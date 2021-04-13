import argparse
import os


parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--g_lr', type=float, default=0.0002, help='lr of generator')
parser.add_argument('--d_lr', type=float, default=0.0002, help='lr of discriminator')
parser.add_argument('--dataset_path', type=str, default='../', help='the folder path of dataset')
parser.add_argument('--input_size', type=int, default=64, help='')
parser.add_argument('--input_channels', type=int, default=3, help='')
parser.add_argument('--gf', type=int, default=128, help='')
parser.add_argument('--df', type=int, default=128, help='')
parser.add_argument('--num_workers', type=int, default=8, help='')
parser.add_argument('--latent_dim', type=int, default=100, help='length of noise')
parser.add_argument('--beta1', type=float, default=0.5, help='')
parser.add_argument('--beta2', type=float, default=0.999, help='')
parser.add_argument('--n_epochs', type=int, default=5, help='')
os.makedirs('./models/',exist_ok=True)
parser.add_argument('--models_path', type=str, default='models/', help='')
os.makedirs('./images/',exist_ok=True)
parser.add_argument('--images_path', type=str, default='images/', help='Intermediate generated image')
parser.add_argument('--train',action='store_true')
parser.add_argument('--test',action='store_true')
parser.add_argument('--k_disc',type=int, help="the number of discriminator's training per batch size", default=1)
parser.add_argument('--early',type=int, help="the number of batch size in early", default=2)
parser.add_argument('--pre_train',action='store_true',help='for train')
parser.add_argument('--g_model_path',type=str, default='./models/g_9500.pth')
parser.add_argument('--d_model_path',type=str, default='./models/d_9500.pth')
parser.add_argument('--n_test',type=int,default=10,help='the number of test images')

args = parser.parse_args()
