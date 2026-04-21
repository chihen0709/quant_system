Self test for quant stock tech tool 

# Quick start
## env setting 
For Debain/Ubuntu Linux distro 
$sudo apt update
$sudo apt install python3-venv python3-pip
## package install
need install `yfinance pandas numpy ta pyTelegramBotAPI playwright scikit-learn scikit-optimize deap backtrader matplotlib schedule` using pip tool 
$ pip install yfinance pandas numpy ta pyTelegramBotAPI playwright scikit-learn scikit-optimize deap backtrader matplotlib schedule
$ playwright install chromium

## start
$ source venv/bin/activate
/*for train ml model to decision*/
$ python train_macro_model.py 
/*Gaussian Regression for optimiztion*/
python optimize_bayesian.py
/*qaunt analysis script for every day*/
$ python quant_pro.py 
or
/*background running*/
$ nohup python quant_pro.py > system_log.txt 2>&1 & 
