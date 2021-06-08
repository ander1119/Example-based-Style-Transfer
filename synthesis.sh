./ebsynth/bin/ebsynth \
    -style inputImage/styleImage/$1.png \
    -guide inputImage/styleImage/G_app/$1.png inputImage/targetImage/G_app/$2.png -weight 4 \
    -guide inputImage/styleImage/segmentation_images/$1.png inputImage/targetImage/segmentation_images/$2.png -weight 7 \
    -guide inputImage/styleImage/G_pos/$1.png inputImage/targetImage/G_pos/$2.png -weight 1 \
    -output result/$1_$2.png