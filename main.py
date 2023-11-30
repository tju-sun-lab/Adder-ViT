import argparse
import datetime
import os
import torch
from torch import optim
from trainer import train_one_epoch, evaluate
# from model_addervit import VisionTransformer
from model_caddervit import VisionTransformer
from data_loader import get_loader


def main(args):

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loader(args)
    model = VisionTransformer(args).to(device)
    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose=True)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()
    
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=test_loader,
                                     device=device,
                                     epoch=epoch)
        
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels")
    parser.add_argument('--data_path', type=str, default='./data/')

    parser.add_argument("--embed_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=4, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2, help="forward multiplier")
    parser.add_argument("--n_layers", type=int, default=6, help="number of encoder layers")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    
    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))