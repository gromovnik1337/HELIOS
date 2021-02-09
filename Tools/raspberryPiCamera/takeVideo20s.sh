#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M%S")

raspivid -t 20000 -o /home/pi/PIcamVideos/$DATE.h264