#!/bin/bash

# -r specifies FPS, of the video.
# -s specifies video size.
# framerate is the rate the next image will show... sort of confusing with -r but
# if you want to show one image long, use this option.
# I might have misunderstood, refer to man page for the options.

ffmpeg \
    -f image2 \
    -framerate 2 \
    -i anim/frame_%d.png \
    -r 30 \
    -crf 5 -pix_fmt yuv420p \
    -vcodec libx264 \
    anim.mp4
