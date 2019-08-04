import torch
from torch import nn, optim
from torchvision.utils import save_image
from torchvision import transforms, datasets
import numpy as np
from tqgm import tqdm
from statistics import mean
from logger import Logger
import time
import os
import cv2 as cv
from torch.autograd import Variable
from data_loader import DataLoader
from utils import predict
from constants import *

#############################################################################
# 顕著性マップの生成アルゴリズム SalGAN の実験用プログラム
# 
# 参考: 
#   https://github.com/imatge-upc/salgan 本家のTheanoによる実装
#   https://github.com/batsa003/salgan/  Pytorchによる実装（主にこちら参考）
# 
# データセット:
#   https://www-users.cs.umn.edu/~qzhao/publications/pdf/webpage_eccv14.pdf
#
#　ToDo:
#   ・loggerの実装
#   ・data_loaderの実装
#   ・Web Page Saliency Dataset への対応
#   ・コード書き切る
#
#############################################################################


# 画像の横の長さ
image_height = INPUT_SIZE[0]
# 画像の縦の長さ
image_width = INPUT_SIZE[1]
# バッチサイズ
batch_size = 32
# Generatorの最初のConvolutionの出力の次元数
ngf = 64
# Discriminatorの最後のConvolutionの出力の次元数
ndf = 64
# 入力画像の次元数
input_nc = 3
# 出力画像の次元数
output_nc = 1
# エポック数
num_epoch = 120
# AdaGradのパラメータ
lr = 0.0003
# 結果の出力先のパス
DIR_TO_OUTPUT = "./generator_output/"
# Generatorの損失重み
alpha = 0.05

#####################################
# Generatorクラスの定義
#####################################
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main
        gconvLayers = [
            # conv1_1
            nn.Conv2d(input_nc, ngf, kernel_size = 1, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            # conv1_2
            nn.Conv2d(ngf, ngf, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            # pool1
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            # conv2_1
            nn.Conv2d(ngf, ngf * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            # conv2_2
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            # pool2
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            # conv3_1
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # conv3_2
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # conv3_3
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # pool3
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            # conv4_1
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv4_2
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv4_3
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # pool4
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            # conv5_1
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv5_2
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv5_3
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv6_1
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv6_2
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv6_3
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # Upsample6
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            # conv7_1
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv7_2
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # conv7_3
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),      
            # upsample7
            nn.Upsample(scale_factor = 2, mode = "nearest"),  
            # conv8_1
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # conv8_2
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # conv8_3
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # upsample8
            nn.Upsample(scale_factor = 2, mode = "nearest"), 
            # conv9_1
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            # conv9_2
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            # upsample9
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            # conv10_1
            nn.Conv2d(ngf * 2, ngf, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            # conv10_2
            nn.Conv2d(ngf, ngf, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True), 
            # final 1*1 convolutional layer with sigmoid non-linerarity
            nn.Conv2d(ngf, output_nc, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(output_nc),        
            nn.Sigmoid()
        ]
        self.convLayers = nn.Sequential(*gconvLayers)

    def forward(self, input):
        X = self.convLayers(input)
        return X

#####################################
# Discriminatorクラスの定義
#####################################
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        dconvLayers = [
            # conv1_1
            nn.Conv2d(input_nc + 1, input_nc, kernel_size = 1, stride = 1, padding = 1),
            nn.BatchNorm2d(input_nc),
            nn.ReLU(inplace = True),
            # conv1_2
            nn.Conv2d(input_nc, ndf / 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ndf / 2),
            nn.ReLU(inplace = True),
            # pool1
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            #conv2_1
            nn.Conv2d(ndf / 2, ndf, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace = True),
            # conv2_2
            nn.Conv2d(ndf, ndf, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace = True),
            # pool2
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            # conv3_1
            nn.Conv2d(ndf, ndf, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace = True),
            # conv3_2
            nn.Conv2d(ndf, ndf, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace = True),
            # pool3
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            
        ]
        fcModules = [
            # fc4 (テンソルの縦横の長さは一回のプーリングで1/2になるので，
            # 現在のテンソルの大きさはプーリング三回で image_height / 2^3  *  image_width / 2^3)
            nn.Linear(ndf * (image_height / 8) * (image_width / 8), 100),
            nn.Tanh(),
            # fc5
            nn.Linear(100, 2),
            nn.Tanh(),
            # fc6
            nn.Linear(2, 1),
            nn.Sigmoid()
        ]
        self.convLayers = nn.Sequential(*dconvLayers)
        self.fcModules = nn.Sequential(*fcModules)
    
    def forward(self, input):
        X = self.convLayers(input) # 畳み込み層
        X = X.view(-1, self.num_flat_features(X)) # [batch_size, ndf * (image_height / 2^3) * (image_width / 2^3)]の二次元テンソルへ変換
        X = self.fcModules(X) # 全結合層
        return X

    # 全結合層へ送るために，[batch_size, num_flat_features]の2次元テンソルへ変換するために各バッチに対応するノードの数を計算
    def num_flat_features(self, input):
        size = input.size()[1:] # バッチサイズの次元以外を抽出
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#####################################
# Vraiableラッピング関数の定義                 
#####################################
def to_variable(x, require_grad = True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, require_grad)


if __name__ == "__main__":

    #####################################
    # ログ管理                    
    #####################################
    logger = Logger("./logs")

    #####################################
    # データの読み込み                    
    #####################################
    dataloader = DataLoader(batch_size)
    num_batch = dataloader.num_batches

    #####################################
    # GeneratorとDiscriminatorを用意                    
    ####################################
    discriminator = Discriminator()
    generator = Generator()

    #####################################
    # GPUの利用の有無を確認               
    #####################################
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
    
    #####################################
    # 損失関数を定義          
    #####################################
    loss_function = nn.BCELoss()

    #####################################
    # 最適化関数を定義                    
    #####################################
    d_optim = optim.Adagrad(discriminator.parameters(), lr = lr)
    g_optim = optim.Adagrad(generator.parameters(), lr = lr)

    #####################################
    # 出力先の確認・時間測定                    
    #####################################
    # 測定開始
    start_time = time.time()
    # ディレクトリがなければ作成
    if not os.path.exists(DIR_TO_OUTPUT):
        os.makedirs(DIR_TO_OUTPUT)
    validation_sample = cv.imread("sample.png")

    #####################################
    # 学習                    
    #####################################
    counter = 0
    for epoch in tqdm(range(1, num_epoch + 1)):
        # GeneratorとDiscriminatorを交互に学習するための管理用変数
        n_updates = 1

        # バッチ内のコストを初期化しておく
        d_cost_avg = 0
        g_cost_avg = 0

        for i in range(batch_size):
            # バッチ内のオリジナル画像と顕著性マップを取得
            (batch_img, batch_map) = dataloader.get_batch()
            # それぞれ計算グラフの情報を保持させる機能を追加（Variableでラッピング）
            batch_img = to_variable(batch_img, require_grad = False) # [batch_size, 3, image_height, image_width]
            batch_map = to_variable(batch_map, require_grad = False) # [batch_size, 1, image_height, image_width]
            # バッチサイズの長さの本物・偽物ラベルを用意する（BCE損失計算用）
            real_labels = to_variable(torch.Tensor(np.ones(batch_size, dtype = float)), require_grad = False)
            fake_labels = to_variable(torch.Tensor(np.zeros(batch_size, dtype = float)), require_grad = False)

            # Discriminatorの学習
            if n_updates % 2 == 1:
                # discriminator.zero_grad()
                # 勾配の初期化
                d_optim.zero_grad()
                # Discriminatorに正解のセットを"1"と判別させるように入力を定義する
                # Discriminatorへの入力はオリジナル画像と顕著性マップを結合させた4次元テンソル
                input_d = torch.cat((batch_img, batch_map), 1) # [batch_size, 3 + 1, image_height, image_width]に変換
                # Discriminatorに正解データを入力し，出力を得る
                outputs = discriminator(input_d).squeeze() # Discriminatorの出力は[batch_size, 1]の2次元配列で出てくるので，1次元配列にする
                # 正解データに対するBCE損失を計算
                d_real_loss = loss_function(outputs, real_labels)
                # 正解データに対する出力の平均値（スコア）を保存
                real_score = outputs.data.mean()
                # Discriminatorの出力の対数を取って合計
                d_loss = torch.sum(torch.log(outputs))
                # コストを加算
                d_cost_avg += d_loss.data[0]
                # 勾配を計算
                d_loss.backward()
                # 収束を監視
                d_loss.register_hook(print)
                # パラメータの更新
                d_optim.step()
                # 学習状況の保存（損失・正解に対するスコア）
                info = {
                    "d_loss": d_loss.data[0],
                    "real_score_mean": real_score
                }
                # ログを残す
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, counter)
            
            # Generatorの学習
            else:
                # generator.zero_grad()
                # 勾配の初期化
                g_optim.zero_grad()
                # 偽の顕著性マップを生成
                fake_map = generator(batch_img)
                # Discriminatorへの入力はオリジナル画像と顕著性マップを結合させた4次元テンソル
                input_d = torch.cat((batch_img, fake_map), 1) # [batch_size, 3 + 1, image_height, image_width]に変換
                # Discriminatorに判別させる
                outputs = discriminator(input_d).squeeze() # Discriminatorの出力は[batch_size, 1]の2次元配列で出てくるので，1次元配列にする
                # 偽データに対する出力の平均値（スコア）を計算する
                fake_score = outputs.data.mean()
                # Generatorの生成した偽マップと正解マップのBCE損失を計算（Content Loss）
                g_content_loss = loss_function(fake_map, batch_map)
                # Discriminatorとの敵対的損失の計算（Adversarial Loss）
                g_adversarial_loss = - torch.log(outputs)
                # Generatorの損失はマップ同士の画像特徴的損失とdiscriminatorとの敵対的損失の重み付き和
                g_loss = torch.sum(g_content_loss + alpha * g_adversarial_loss)
                # コストを保存
                g_cost_avg += g_loss.data[0]
                # 勾配を計算
                g_loss.backward()
                # パラメータの更新
                g_optim.step()
                # 学習状況の保存
                info = {
                    "g_loss": g_loss.data[0],
                    "fake_score_mean": fake_score,
                }
                # ログを残す
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, counter)
            
            # 学習先の切り替え
            n_updates += 1

            # 学習状況の報告
            if (i + 1) % 100 == 0:
                print("Epoch [%d/%d], Step [%d/%d], d_loss: %.4f, D(x): %2.f, D(G(x)): %.2f, time: %4.4f" %(epoch, num_epoch, i + 1, num_batch, d_loss.data[0], real_score, fake_score, time.time() - start_time))
            
            # カウントアップ
            counter += 1
        
        # バッチ内平均コストを計算
        d_cost_avg /= num_batch
        g_cost_avg /= num_batch

        # 3回に一回モデルの重みを保存する
        if (epoch + 1) % 3 == 0:
            print("Epoch: ", epoch, "train_loss-> ", (d_cost_avg, g_cost_avg))
            torch.save(generator.state_dict(), "./generator.pkl")
            torch.save(discriminator.state_dict(), "./discriminator.pkl")

        # サンプルで顕著性マップを生成してみる
        predict(generator, validation_sample, epoch, DIR_TO_OUTPUT)
    
    # モデルの重みの保存
    torch.save(generator.state_dict(), "./generator.pkl")
    torch.save(discriminator.state_dict(), "./discriminator.pkl")

    print("Finished trainning")
    










    


        
