#!/usr/bin/env bash

# remove border, leaving only the image of the eye
convert -fuzz 10% -trim +repage -resize 256x256 -gravity center -background black -extent 256x256 -equalize 10_left.jpeg 10_left_conv.jpeg
