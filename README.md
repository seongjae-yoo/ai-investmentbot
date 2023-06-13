# AI-InvestmentBot
자동주식투자프로그램

* commit 기록 공간에 실행결과 캡쳐 사진들과 함께 올려 두었습니다.

링크 클릭 - [https://github.com/SEONGJAE-YOO/AI-InvestmentBot/commits/main](https://github.com/SEONGJAE-YOO/AI-InvestmentBot/commits/main)

# "How to Set the Environment"

## Why Miniconda3 is Superior to Anaconda

Miniconda3 has several key advantages over Anaconda, making it a preferred choice for many users. Anaconda comes pre-packaged with a large number of libraries, making it bulky and potentially slowing down system performance. In contrast, Miniconda3 provides a lightweight, clean installation of Python and Conda, devoid of any superfluous packages.

Miniconda3's minimalist design allows users more control over their working environment. You get to choose which libraries to install, which means you only have what you need and nothing more, resulting in a more efficient and optimized workspace.

Therefore, as an initial step, this system was established by constructing a Miniconda3 working environment.


# Stock Data Collection Methods

### This research harnessed the Kiwoom Securities OpenAPI library, a Component Object Model (COM)-based library that exclusively supports a Python 32-bit environment, for gathering daily stock data from around 2,546 stocks listed on KOSPI, KOSDAQ, and KONEX, courtesy of Kiwoom Securities.

## 1. set CONDA_FORCE_32BIT=1 && conda create -n py37_32 python=3.7.13


## 2. pip install -r requirements_py37_32.txt

## 3. In a 32-bit virtual environment, entering the command 'python batch_generator.py' in CMD will automatically create four files in the 'bat' folder: ai_filter.bat, collector.bat, simul_run.bat, and trader.bat.

## Example 
```cmd
(py37_32) C:\Users\SeongJae-Yoo\AI-InvestmentBot>python batch_generator.py
배치파일을 성공적으로 생성하였습니다.
```


### Setting Up Automatic Login
After manually logging in via the Open API connection in KOAStudioSA, right-click on the Open API icon displayed on the taskbar in the bottom right corner.
When the account password input window appears, automatic login can be configured by selecting the "Save Account Password" option.
Enter the account and account password you wish to use and click on the 'Register' button to save.
Lastly, select the AUTO checkbox.


## 4. Executing the collector.bat file automatically collects stock data.


## Method for creating a 64-bit virtual environment.

## 1. conda create -n py37_64 python=3.7.13 

## 2. pip install -r requirements_py37_64.txt  


### Backtesting experiments can be conducted in a 64-bit virtual environment, and Deep learning models can be employed for forecasting stock trends.




# Backtesting Methods

## Backtesting period starts from January 3, 2022, and simulations can be conducted up to the data collection period

The simulator_func_mysql.py file is executed.
Enter the following command in the terminal within the cmd environment:
python simulator.py
Upon entering the command above, you will see the execution results as shown in the following picture.

![cmd](https://github.com/SEONGJAE-YOO/AI-InvestmentBot/blob/main/Image/20230510_213012_1.png)


When prompted with the sentence "Please enter the algorithm number you want to simulate:", if you enter 29, you can simulate backtesting the CNN Attention BiLSTM model. After entering the number 29, you can choose whether to initialize the database (y or n) for the results. If you enter 'y', it will be possible to run from the beginning of the set period. If you enter 'n', you can perform backtesting simulation from the last executed period.


![cmd](https://github.com/SEONGJAE-YOO/AI-InvestmentBot/blob/main/Image/20230510_213012_2.png)

When prompted with the sentence "Please enter the algorithm number you want to simulate:", if you enter 30, you can simulate backtesting the BiGRU CNN BiLSTM Attention model. The database initialization options (y or n) are the same as previously mentioned.

When prompted with the sentence "Please enter the algorithm number you want to simulate:", if you enter 31, you can simulate backtesting the BiLSTM Attention CNN model. The database initialization options (y or n) are the same as previously mentioned.

When prompted with the sentence "Please enter the algorithm number you want to simulate:", if you enter 32, you can simulate backtesting the CNN BiLSTM Attention Model. The database initialization options (y or n) are the same as previously mentioned.






## Result of Backtesting using CNN Attention BiLSTM model

### After simulating number 29, you can check the profit results by entering the following command in the database.

```sql
select code, code_name, rate, purchase_price,holding_amount,item_total_purchase,buy_date,sell_date
from simulator29.all_item_db
```

##Result of Backtesting using BiGRU CNN BiLSTM Attention model

### After simulating number 30, you can check the profit results by entering the following command in the database.

```sql
select code, code_name, rate, purchase_price,holding_amount,item_total_purchase,buy_date,sell_date
from simulator30.all_item_db
```

## Result of Backtesting using BiLSTM Attention CNN model

### After simulating number 31, you can check the profit results by entering the following command in the database.

```sql
select code, code_name, rate, purchase_price,holding_amount,item_total_purchase,buy_date,sell_date
from simulator31.all_item_db
```

## Result of Backtesting using CNN BiLSTM Attention Model

### After simulating number 32, you can check the profit results by entering the following command in the database.

```sql
select code, code_name, rate, purchase_price,holding_amount,item_total_purchase,buy_date,sell_date
from simulator32.all_item_db
```


# Supplementary Materials: 
The outcomes of the comparative analysis, which encompassed 
eight deep learning models utilized in this research, along with the findings from the 
WandB Sweep experiment, are presented on the subsequent page.

In the master's thesis, Celltrion Healthcare and Samsung Electronics stocks, the deep learning model experiment results can be found on the link page below.

[https://wandb.ai/seongjae-yoo/projects](https://wandb.ai/seongjae-yoo/projects)


# error 해결방안
## 1. python demo.py 파일 실행 할때 oserror symbolic link privilege not held 다음과 같이 에러 나면 vscode 터미널 관리자로 열어서 해결하세요  (아래 링크 참고 후 에러 해결하세요)
## https://parodev.tistory.com/47   참고