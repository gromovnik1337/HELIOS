#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M%S")

raspistill -t 1 -o /home/pi/PIcamPictures/$DATE.jpg
