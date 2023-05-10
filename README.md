# AI-InvestmentBot
자동주식투자프로그램

* commit 기록 공간에 실행결과 캡쳐 사진들과 함께 올려 두었습니다.

링크 클릭 - [https://github.com/SEONGJAE-YOO/AI-InvestmentBot/commits/main](https://github.com/SEONGJAE-YOO/AI-InvestmentBot/commits/main)

# 1. Miniconda3 설치 
# 2. 가상환경 conda create -n py37_64 python=3.7.13  설치
# 3. pip install -r requirements_py37_64.txt  설치


# error 해결방안
## 1. python demo.py 파일 실행 할때 oserror symbolic link privilege not held 다음과 같이 에러 나면 vscode 터미널 관리자로 열어서 해결하세요  (아래 링크 참고 후 에러 해결하세요)
## https://parodev.tistory.com/47   참고


# backtesting method 

## Backtesting period starts from January 3, 2022, and simulations can be conducted up to the data collection period

The simulator_func_mysql.py file is executed.
Enter the following command in the terminal within the cmd environment:
python simulator.py
Upon entering the command above, you will see the execution results as shown in the following picture.

![cmd](image/20230510_213012_1.png)


When prompted with the sentence "Please enter the algorithm number you want to simulate:", if you enter 29, you can simulate backtesting the CNN Attention BiLSTM model. After entering the number 29, you can choose whether to initialize the database (y or n) for the results. If you enter 'y', it will be possible to run from the beginning of the set period. If you enter 'n', you can perform backtesting simulation from the last executed period.


![cmd](image/20230510_213012_2.png)

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
