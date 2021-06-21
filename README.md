

## 前置步驟

1. 請 cd 到 Example-Based-Synthesis-of-Stylized-Facial-Animations 這個資料夾。

2. 將 style exemplar 放到 inputImage/sytleImage 資料夾當中。

3. 將 target image 放到 inputImage/targetImage 當中。

4. 安裝相對應之作業系統的 ebsynth ，並將 ebsynth 的資料夾放到當前資料夾中，用於 synthesis 步驟。

   ```shell
   ## 例如：在 macos 上
   git clone https://github.com/jamriska/ebsynth
   sh build-macos-cpu_only.sh
   ```

## 直接使用 run.sh 以及 synthesis.sh

```shell
## 將欲合成之 style image 以及 target image 的檔名放入以下的 [image name] 中
## 此步驟會將 Gseg, G_pos, G_app 都完成。
sh run.sh [style image name] [target image name]

## 此步驟會將所有階段性圖片合成出 output image。
## 後面的三個數字代表 weight，可使用預設是 10, 15, 5。
## 請注意 synthesis.sh 中的 ebsynth 執行路徑是 ./ebsynth/bin/ebsynth
sh synthesis.sh [style image name] [target image name] 10 15 5
```



## 以下將介紹每個詳細步驟的使用方式

### 製作 Segmentation Guide

```shell

## 以下指令會將所有放在 inputImage/styleImage 的圖片都做出 Segmentation Guide 的階段圖片。
python3 ./code/Gseg/segment.py \
    --input_dir ./inputImage/styleImage  
    --conf_threshold 0.99

## 以下指令會將所有放在 inputImage/targetImage 的圖片都做出 Segmentation Guide 的階段圖片。
python3 ./code/Gseg/segment.py  
    --input_dir ./inputImage/targetImage 
    --conf_threshold 0.99
```

### 製作 Positional Guide

```shell
## 選擇欲製作 position guide 的 style 與 target 圖片，放入底下的 [image name] 中。

python3 code/Gpos/mlsAffineMorpher.py \
    --style_dir ./inputImage/styleImage \
    --style_img [style image name].png  \
    --input_dir ./inputImage/targetImage \
    --input_img [target image name].png
```

### 製作 Appearance Guide

```shell
## 選擇欲製作  guide 的 style 與 target 圖片，放入底下的 [image name] 中。

python3 code/Gapp/generateGapp.py \
    --style_dir ./inputImage/styleImage \
    --style_img [style image name].png  \
    --input_dir ./inputImage/targetImage \
    --input_img [target image name].png
```

###  Synthesize 以上階段性圖片。

```shell
## 選擇欲合成的 style 與 target 圖片，放入底下的 [image name] 中。
## 預設的 weight 依序是 10, 15, 5
## 產生的圖片將會放在 output 資料夾中。
./ebsynth/bin/ebsynth \
    -style inputImage/styleImage/[style image name].png \
    -guide inputImage/styleImage/G_app/[style image name].png inputImage/targetImage/G_app/[target image name].png -weight 10 \
    -guide inputImage/styleImage/segmentation_images/[style image name].png inputImage/targetImage/segmentation_images/[target image name].png -weight 15 \
    -guide inputImage/styleImage/G_pos/[style image name].png
    inputImage/targetImage/G_pos/[target image name].png -weight 5 \
    -output result/[style image name]_[target image name].png

```



