./ebsynth/bin/ebsynth -style inputImage/styleImage/style1.png -guide inputImage/styleImage/G_app/style1.png inputImage/targetImage/G_app/target1.png -weight 2.0 -guide inputImage/styleImage/segmentation_images/style1.png inputImage/targetImage/segmentation_images/target1.png -weight 1.5 -guide inputImage/styleImage/G_pos/style1.png inputImage/targetImage/G_pos/target1.png -weight 1.5 -output output.png