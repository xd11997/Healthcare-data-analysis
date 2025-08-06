import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
#---------------------------------------------------------------------
#第一步：计算PDC
# 1. 读取CSV文件，换成你实际文件名
file_path = 'prescription_refill.csv'
df = pd.read_csv(file_path)

# 2. 转换日期格式
df['fill_date'] = pd.to_datetime(df['fill_date'])

# 3.1 更改数据集，通过缩短处方增加波动性
np.random.seed(42)
mask = np.random.rand(len(df)) < 0.7
    # 生成布尔掩码mask，np.random.rand()指生成长度为（）的随机数组，目标是选出40%的行
df.loc[mask, 'days_supply'] = (df.loc[mask, 'days_supply']*np.random.uniform(0.3, 0.6, size=mask.sum())).astype(int)
    # df.loc是用来按标签或布尔数组定位并访问 DataFrame 的特定行/列的方式
    # df.loc[mask, 'days_supply']指从原表中取出所有被标记为 True 的行，选取 days_supply 这一列的值
    # np.random.uniform(0.7, 0.9, size=mask.sum()) → 为这些行生成一个与它们数量一样的随机数组，每个数介于 0.7 到 0.9 之间
    # uniform指均匀分布
    # .astype(int) → 转换为整数

# 3.2 计算每个患者每次买药的覆盖结束日期（fill_date + days_supply -1）
# pandas.to_timedelta() 是用来把数字（比如整数）变成“时间跨度”的函数，unit='D' 指要转换的数字是“天（Days）”
df['end_date'] = df['fill_date'] + pd.to_timedelta(df['days_supply'] - 1, unit='D')

# 4. 定义函数合并区间
def merge_intervals(intervals):
    """
    intervals: list of (start_date, end_date) tuples（元组）, sorted by start_date
    返回合并后的不重叠区间列表
    """
    merged = [] #新建一个空列表 merged，用于存放最终合并后的时间区间
    for current in intervals:
            if not merged:
                merged.append(current) #append指向列表内增加元素
            else:
                #以下两行目的为将两个区间拆开，提取出它们的起止时间，以便合并或比较
                prev_start, prev_end = merged[-1] #merged[-1]指上一个已合并区间（列表索引）
                curr_start, curr_end = current #current指当前正要处理的区间
                if curr_start <= prev_end + timedelta(days=1): # 重叠或相邻
                    # 合并区间
                    merged[-1] = (prev_start, max(prev_end, curr_end))
                else:
                    merged.append(current)
    return merged

# 5. 计算每个患者的PDC
pdc_list = [] # 新建一个空列表 pdc_list，用来保存每个患者的 PDC 计算结果

for patient_id, group in df.groupby('patient_id'): #df.groupby('patient_id')指根据patient_id分组，对每个组分别处理
    # 按fill_date排序
    intervals = list(zip(group['fill_date'], group['end_date'])) #用 zip() 把 fill_date 和 end_date 组合成区间 (start_date, end_date)
    intervals.sort(key=lambda x: x[0])

    merged_intervals = merge_intervals(intervals)

    # 计算覆盖天数总和
    covered_days = sum((end - start).days + 1 for start, end in merged_intervals)
        #遍历合并后的所有区间 (start, end)，用 end - start 计算每段区间长度（是 timedelta）
        #.days + 1：加 1 是因为两端都算在内（比如 1月1日到1月1日应该是 1 天，而不是 0）。

    # 观察期总天数（从最早fill_date到最晚end_date）
    observation_start = min(group['fill_date'])
    observation_end = max(group['end_date'])
    observation_days = (observation_end - observation_start).days + 1

    pdc = covered_days / observation_days if observation_days > 0 else 0

    pdc_list.append({
        'patient_id': patient_id,
        'covered_days': covered_days,
        'observation_days': observation_days,
        'pdc_ratio': round(pdc, 4)
    })

# 6. 转成DataFrame输出
pdc_df = pd.DataFrame(pdc_list)

#print(pdc_df['pdc_ratio'].describe())
#---------------------------------------------------------------------
#第二步：构建训练集：特征与标签，保存新数据集
file1_path = 'patient_demographics.csv'
patient_df = pd.read_csv(file1_path)
file2_path = 'insurance_claims.csv'
insurance_df = pd.read_csv(file2_path)
# 1. 合并 PDC 比例以及保险理赔数据到患者人口统计表
train_df = patient_df.merge(pdc_df[['patient_id', 'pdc_ratio']], on='patient_id', how='left')
train_df = train_df.merge(insurance_df, on='patient_id', how='left')
# 2. 创建二分类标签列：pdc_ratio < 0.80 视为 non-adherent（label = 1）
train_df['label_non_adherent'] = (train_df['pdc_ratio'] < 0.9).astype(int)
# 3.1 制造缺失值
missing_indices1 = np.random.choice(train_df.index, size=10, replace=False) #随机选10个索引，replace=False 指不放回抽样（抽到一次之后就不能再抽到）
missing_indices2 = np.random.choice(train_df.index, size=8, replace=False) #随机选8个索引
train_df.loc[missing_indices1, 'age'] = np.nan
train_df.loc[missing_indices2, 'copay_amount'] = np.nan

# 4. 保存数据集
train_df.to_csv('patient_demographics_total.csv', index=False) #不要把 DataFrame 的索引列写进文件中，否则保存的 CSV 文件里，会多出一列数字 0, 1, 2...作为行号，可能会影响后续加载
# print(train_df.head())
# print(train_df['label_non_adherent'].value_counts()) #统计某一列中每个值的数量
#---------------------------------------------------------------------
#第三步：logistic回归
# 1. 缺失值处理
# 1.1 查看缺失值数量
# missing_counts = train_df.isnull().sum()
# plt.figure(figsize=(8, 5))
# missing_counts.sort_values(ascending=False).plot(kind='bar')
# plt.title('Missing Values per Feature')
# plt.xlabel('Feature')
# plt.ylabel('Number of Missing Values')
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

# 1.2 使用'Missing' 填补类别型缺失值
logistic_df = pd.read_csv('patient_demographics_total.csv')
logistic_df['chronic_disease'] = logistic_df['chronic_disease'].fillna('Missing')
# print(logistic_df['chronic_disease'].value_counts(dropna=False))

# 1.3 使用中位数填补数值型缺失值（对偏态数据更robust）
# 中位数填充（对偏态数据更鲁棒）
logistic_df['age'] = logistic_df['age'].fillna(logistic_df['age'].median())
logistic_df['copay_amount'] = logistic_df['copay_amount'].fillna(logistic_df['copay_amount'].median())
# print(logistic_df[['age', 'copay_amount']].describe())

# 2. 特征选择与处理
features = ['age', 'gender','copay_amount','comorbidity_count','chronic_disease', 'zip_poverty_rate', 'insurance_type']
target = 'label_non_adherent'
# 类别型变量编码『one-hot encoding』
logistic_df = pd.get_dummies(logistic_df, columns=['gender', 'chronic_disease', 'insurance_type'], drop_first=True) #drop_first=True为了避免多重共线性
# 构造 X 和 y
X = logistic_df[[col for col in logistic_df.columns if col in features or
               col.startswith('gender_') or
               col.startswith('chronic_disease_') or
               col.startswith('insurance_type_')]]
y = logistic_df[target]

# 3. 标准化数值变量（类别型变量已经独热编码成0/1，不需要标准化，标准化只针对连续变量）
scaler = StandardScaler() #StandardScaler 将数据转成均值为0、标准差为1的分布
X_scaled = X.copy()
X_scaled[['age', 'copay_amount', 'comorbidity_count', 'zip_poverty_rate']] = scaler.fit_transform(
    X_scaled[['age', 'copay_amount', 'comorbidity_count', 'zip_poverty_rate']]
)

# 4. 拆分训练集和测试集（80%用于训练，20%留作测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 5.1 训练sklearn逻辑回归模型&评估
log_reg = LogisticRegression(max_iter=1000) #用训练集数据 X_train, y_train 训练逻辑回归模型
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5.2 用 statsmodels 输出模型详细统计信息和变量解释
# 找出所有bool类型列
bool_cols = logistic_df.select_dtypes(include=['bool']).columns

# 转换为int类型
logistic_df[bool_cols] = logistic_df[bool_cols].astype(int)
# 加常数项
X_const = sm.add_constant(X_scaled)
X_const = X_const.astype(float) #确保所有数据都是浮点数！

#调用Logit拟合
model = sm.Logit(y, X_const).fit()
print(model.summary())
#计算Odds Ratio (OR)，表示某变量每单位增加，事件（非依从）发生的比率变化倍数
odds_ratios = np.exp(model.params)
print("\nOdds Ratios:\n", odds_ratios)







