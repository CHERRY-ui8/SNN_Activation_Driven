"""
CIFAR10 Activation-Driven SNN Baseline Training (Task 1 Core)
Use default ATan surrogate gradient function to validate the feasibility of the baseline network
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler  # mixed precision training
from spikingjelly.activation_based.surrogate import ATan, Sigmoid  # default surrogate gradient function
from spikingjelly.activation_based import functional

# add project root to Python path to ensure importing src module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# import custom modules
from src.models.model_cifar10_csnn import CIFAR10VGG16, CIFAR10ResNet18
from src.data.data_cifar10 import load_cifar10
from src.utils.utils import get_device, init_tensorboard, save_checkpoint, calculate_metrics
from src.surrogate.surrogate_custom import SigmoidPrimeSurrogate, EsserSurrogate, SuperSpikeSurrogate

def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='CIFAR10 Baseline SNN Training (Activation-Driven)')
    # device parameters
    parser.add_argument('--device', default='cuda:0', type=str, help='device ID (e.g. cuda:0, cpu)')
    # model parameters
    parser.add_argument('--model', default='resnet18', type=str, help='model name (vgg16, resnet18)')
    parser.add_argument('--surrogate', default='Sigmoid', type=str, help='surrogate function (ATan, SigmoidPrime, Esser, SuperSpike)')
    parser.add_argument('--surrogate_beta', default=4.0, type=float, help='beta parameter for surrogate function')
    parser.add_argument('--T', default=4, type=int, help='time step (length of spike sequence)')
    # training hyperparameters
    parser.add_argument('--epochs', default=64, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer (adam, adamw, sgd)')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning rate (lower bound of cosine annealing)')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay (L2 regularization)')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum (only used when optimizer=sgd)')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping threshold (0 means no clipping)')
    parser.add_argument('--patience', default=5, type=int, help='early stopping patience (stop if no improvement for consecutive epochs, 0 means no early stopping)')
    parser.add_argument('--no_amp', action='store_true', help='disable AMP mixed precision (AMP is only enabled on CUDA by default)')
    # data and log
    parser.add_argument('--data_dir', default='./datasets/CIFAR10', type=str, help='CIFAR10 dataset directory')
    parser.add_argument('--log_dir', default='./logs/cifar10', type=str, help='log directory')
    parser.add_argument('--resume', default='', type=str, help='checkpoint path for resuming training (optional)')
    return parser.parse_args()

def train_one_epoch(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter
) -> tuple[float, float]:
    """train one epoch"""
    net.train()  # training mode
    total_train_loss = 0.0
    total_train_acc = 0.0
    total_samples = 0.0
    start_time = time.time()

    for batch_idx, (img, label) in enumerate(train_loader):
        # move data to device
        img, label = img.to(device), label.to(device)
        batch_size = img.shape[0]
        total_samples += batch_size

        # zero gradients
        optimizer.zero_grad()
        # mixed precision training (accelerate training, reduce memory usage)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and (not args.no_amp))):
            # forward propagation: calculate average firing rate
            out_fr = net(img)
            # activation driven loss: MSE loss (minimize the difference between the firing rate and the target one-hot vector)
            loss = F.cross_entropy(out_fr, label)

        # backward propagation and optimization
        scaler.scale(loss).backward()
        # gradient clipping (prevent gradient explosion, improve training stability)
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)  # cancel scaling for gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)  # optimizer update
        scaler.update()  # update scaler

        # calculate current batch accuracy and average loss
        batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
        total_train_acc += batch_acc * batch_size
        # batch_avg_loss is already mean loss over batch; weight it by batch_size to get per-sample average later
        total_train_loss += batch_avg_loss * batch_size

        # reset neuron state (avoid state accumulation after multi-step training)
        functional.reset_net(net)

        # print batch information (print every 10 batches)
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Batch Loss: {batch_avg_loss:.4f} | Batch Acc: {batch_acc:.4f}")

    # calculate current epoch average loss and accuracy
    avg_train_loss = total_train_loss / total_samples
    avg_train_acc = total_train_acc / total_samples
    train_time = time.time() - start_time

    # record TensorBoard logs
    writer.add_scalar('Train/Avg_Loss', avg_train_loss, epoch)
    writer.add_scalar('Train/Avg_Acc', avg_train_acc, epoch)

    # print epoch summary
    print(f"\nEpoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {avg_train_acc:.4f} | Time: {train_time:.2f}s")

    return avg_train_loss, avg_train_acc

def test_one_epoch(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter
) -> tuple[float, float]:
    """test one epoch (no gradient calculation)"""
    net.eval()  # evaluation mode
    total_test_loss = 0.0
    total_test_acc = 0.0
    total_samples = 0.0

    with torch.no_grad():  # disable gradient calculation
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            batch_size = img.shape[0]
            total_samples += batch_size

            # forward propagation
            out_fr = net(img)
            # calculate loss
            loss = F.cross_entropy(out_fr, label)

            # calculate batch metrics
            batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
            total_test_acc += batch_acc * batch_size
            total_test_loss += batch_avg_loss * batch_size

            # reset neuron state
            functional.reset_net(net)

    # calculate average metrics
    avg_test_loss = total_test_loss / total_samples
    avg_test_acc = total_test_acc / total_samples

    # record TensorBoard logs
    writer.add_scalar('Test/Avg_Loss', avg_test_loss, epoch)
    writer.add_scalar('Test/Avg_Acc', avg_test_acc, epoch)

    # print test summary
    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f}\n")

    return avg_test_loss, avg_test_acc

def main(args: argparse.Namespace):
    # 1. initialize device
    device = get_device(args.device)

    # 2. load data
    print("\n=== load CIFAR10 dataset ===")
    train_loader, test_loader = load_cifar10(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )

    # 3. initialize model (default use ATan surrogate gradient function)
    print("\n=== initialize model ===")
    if args.surrogate == 'ATan':
        surrogate_func = ATan()
    elif args.surrogate == 'Sigmoid':
        surrogate_func = Sigmoid(alpha=args.surrogate_beta)
    elif args.surrogate == 'SigmoidPrime':
        surrogate_func = SigmoidPrimeSurrogate(beta=args.surrogate_beta)
    elif args.surrogate == 'Esser':
        surrogate_func = EsserSurrogate(beta=args.surrogate_beta)
    elif args.surrogate == 'SuperSpike':
        surrogate_func = SuperSpikeSurrogate(beta=args.surrogate_beta)
    else:
        raise ValueError(f"Invalid surrogate function: {args.surrogate}")
    
    if args.model == 'vgg16':
        net = CIFAR10VGG16(
            T=args.T,
            surrogate_func=surrogate_func
        ).to(device)
    elif args.model == 'resnet18':
        net = CIFAR10ResNet18(
            T=args.T,
            surrogate_func=surrogate_func
        ).to(device)
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    # print model basic information (avoid using repr to avoid recursive error)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"model configuration: T={args.T}, surrogate={args.surrogate}, surrogate_beta={args.surrogate_beta}")
    print(f"model parameters: total={total_params:,}, trainable={trainable_params:,}")

    # 4. initialize optimizer and learning rate scheduler
    opt_name = args.optimizer.lower()
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer} (expected: adam, adamw, sgd)")
    # cosine annealing learning rate scheduler (decrease learning rate with epoch, but set minimum learning rate to avoid too small)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # 5. initialize mixed precision training scaler
    scaler = GradScaler(enabled=(device.type == 'cuda' and (not args.no_amp)))

    # 6. resume training (if specified checkpoint)
    start_epoch = 0
    max_test_acc = 0.0
    if args.resume != '':
        from src.utils.utils import load_checkpoint
        checkpoint = load_checkpoint(
            net=net,
            optimizer=optimizer,
            checkpoint_path=args.resume,
            device=device
        )
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    # 7. initialize TensorBoard logs
    writer = init_tensorboard(
        log_dir=args.log_dir
        + f'_{args.model}_{args.surrogate}_beta{args.surrogate_beta}_T{args.T}'
        + f'_{opt_name}_lr{args.lr}_wd{args.weight_decay}'
    )

    # 8. start training loop
    print("\n=== start training ===")
    # early stopping mechanism variables
    best_epoch = start_epoch
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        # train one epoch
        train_loss, train_acc = train_one_epoch(
            net=net,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer
        )

        # test one epoch
        test_loss, test_acc = test_one_epoch(
            net=net,
            test_loader=test_loader,
            device=device,
            epoch=epoch,
            writer=writer
        )

        # update learning rate
        lr_scheduler.step()
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # save checkpoint (check if it is the best model)
        is_best = test_acc > max_test_acc
        if is_best:
            max_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0  # reset patience counter
        else:
            patience_counter += 1
        
        save_checkpoint(
            net=net,
            optimizer=optimizer,
            epoch=epoch,
            max_test_acc=max_test_acc,
            save_path=os.path.join(args.log_dir, 'checkpoints'),
            is_best=is_best
        )
        
        # early stopping check
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nearly stopping triggered: no improvement for consecutive {args.patience} epochs")
            print(f"best model at epoch {best_epoch+1}, test accuracy: {max_test_acc:.4f}")
            break

    # training end
    print(f"\n=== training end ===")
    print(f"max test accuracy: {max_test_acc:.4f}")
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
