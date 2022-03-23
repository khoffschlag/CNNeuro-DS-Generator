# CNNeuro-DS-Generator
Dataset generator for the CNNeuro project

# Installation on Ubuntu 21.10
1. Clone repository
> git clone https://github.com/khoffschlag/CNNeuro-DS-Generator.git
2. Enter the new directory
> cd CNNeuro-DS-Generator
3. Install python3.9 and pip3
> sudo apt update && sudo apt install python3.9 python3-pip
4. Install the required python packages
> pip3 install -r requirements.txt
5. You're ready to try out the example code or write your own dataset generation script!

# Troubleshooting

## We smoothed our data with 8mm FWHM but the atrophy transformator wants a sigma value. What can I do?

We can convert the given FWHM smoothing value into a corresponding sigma value.

Use the following equation:
sigma = <sup>FWHM value</sup>&frasl;<sub>2.35482004503</sub>

In your case: sigma = <sup>FWHM value</sup>&frasl;<sub>2.35482004503</sub> = <sup>8</sup>&frasl;<sub>2.35482004503 = 3.397287201153446
  
It's probably sufficient here if you round up to 3.4
