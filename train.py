import os.path

import cv2
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import config
from dataset import DatasetLoader
from utils import save_checkpoint, load_checkpoint
import torch.nn as nn
from tqdm import tqdm
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    A_reals = 0
    A_fakes = 0

    loop = tqdm(loader, leave=True)

    for idx, (B, A) in enumerate(loop):
        B = B.to(config.DEVICE)
        A = A.to(config.DEVICE)

        # Train Discriminators, A and B
        with torch.cuda.amp.autocast():
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # put it together
            D_loss = (D_A_loss + D_B_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators, A and B
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            # cycle loss
            cycler_B = gen_B(fake_A)
            cycler_A = gen_A(fake_B)
            cycle_B_loss = L1(B, cycler_B)
            cycle_A_loss = L1(A, cycler_A)

            # identity loss
            identity_B = gen_B(B)
            identity_A = gen_B(A)
            identity_B_loss = L1(B, identity_B)
            identity_A_loss = L1(A, identity_A)

            # add all together
            G_loss = (
                loss_G_B +
                loss_G_A +
                cycle_B_loss * config.LAMBDA_CYCLE +
                cycle_A_loss * config.LAMBDA_CYCLE +
                identity_B_loss * config.LAMBDA_IDENTITY +
                identity_A_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_A * 0.5 + 0.5, f"saved_images/A_{idx}.png")
            save_image(fake_B * 0.5 + 0.5, f"saved_images/B_{idx}.png")
        loop.set_postfix(A_real=A_reals/(idx+1), A_fake=A_fakes/(idx+1))


def main():
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, gen_A, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_B, gen_B, opt_disc, config.LEARNING_RATE
        )

    dataset = DatasetLoader(
        root_dir=os.path.join(config.DATA_FOLDER, config.DATASET_FOLDER_NAME),
        transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)


if __name__ == '__main__':
    main()
