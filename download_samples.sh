#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download LibriVox public domain audiobook samples
echo "Downloading LibriVox samples..."
curl -L "https://ia800604.us.archive.org/23/items/collected_lovecraft_0810_librivox/halloween_providence_lovecraft.mp3" -o data/lovecraft_sample.mp3
curl -L "https://ia801406.us.archive.org/8/items/short_story_01_0811_librivox/shortstory001_01_andersen_64kb.mp3" -o data/andersen_shortstory.mp3
curl -L "https://ia600302.us.archive.org/25/items/mark_twain_tom_sawyer_ch01-02_librivox/tomsawyer_01_twain_64kb.mp3" -o data/twain_tomsawyer.mp3

echo "Downloads complete!"
echo "Available audio samples:"
ls -lh data/*.mp3
