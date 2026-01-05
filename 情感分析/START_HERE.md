# 🌆 上海CityWalk打卡点情感分析系统 - 使用指南

## 📌 快速导航

- **想立即开始？** → 跳转到 [运行代码](#-立即运行代码)
- **需要详细说明？** → 查看 [使用说明.md](使用说明.md)
- **想要代码示例？** → 查看 [example_usage.py](example_usage.py)
- **要完整文档？** → 查看 [README.md](README.md)
- **项目信息？** → 查看 [项目完成总结.txt](项目完成总结.txt)

---

## 🚀 立即运行代码

### 最简单的方式（推荐）

#### **Windows用户：**
双击打开以下任意一个文件：
1. `快速启动.py` 
2. `run.bat`

#### **Mac/Linux用户或命令行：**
```bash
cd c:\Users\27885\Desktop\citywalk\情感分析
python run_analysis.py
```

### 代码将自动：
1. ✅ 加载示例数据
2. ✅ 提取25个打卡点
3. ✅ 执行情感分析
4. ✅ 生成结果文件：
   - `citywalk_analysis_results.csv` （数据表格）
   - `citywalk_analysis_results.png` （可视化图表）

---

## 📊 完整的Python代码示例

### 核心代码（可直接使用）

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# 1. 情感分析器类
class SentimentAnalyzer:
    def __init__(self):
        # 积极词汇权重
        self.positive_words = {
            '很棒': 0.9, '很好': 0.85, '很美': 0.85, '值得': 0.85,
            '推荐': 0.9, '喜欢': 0.85, '满意': 0.8, '开心': 0.85,
            '舒服': 0.8, '优雅': 0.85, '特色': 0.75, '有趣': 0.85,
            '完美': 0.95, '精妙': 0.85, '精致': 0.8, '古朴': 0.75,
            '独特': 0.75, '创意': 0.8, '艺术': 0.75, '文化': 0.7,
            '安静': 0.75, '清幽': 0.8, '宁静': 0.8, '祥和': 0.85,
            '浪漫': 0.85, '繁华': 0.7, '活力': 0.75, '欢乐': 0.85,
        }
        
        # 否定词汇权重
        self.negative_words = {
            '很差': 0.15, '不好': 0.2, '很丑': 0.1, '讨厌': 0.05,
            '失望': 0.25, '后悔': 0.15, '浪费': 0.2, '不满': 0.25,
            '拥挤': 0.3, '排队': 0.35, '贵': 0.35, '缺少': 0.35,
            '冷清': 0.35, '破旧': 0.2, '不方便': 0.3, '疲惫': 0.3,
        }
        
        self.negation_words = {'不', '没', '无', '别'}
    
    def analyze(self, text):
        """分析情感得分 (0-1分)"""
        if not text:
            return 0.5
        
        # 提取中文词汇
        words = re.findall(r'[\u4e00-\u9fa5]+', text)
        
        pos_score = 0
        neg_score = 0
        
        for i, word in enumerate(words):
            # 检查积极词
            if word in self.positive_words:
                score = self.positive_words[word]
                # 处理否定（如"不好"）
                if i > 0 and words[i-1] in self.negation_words:
                    neg_score += score
                else:
                    pos_score += score
            
            # 检查否定词
            elif word in self.negative_words:
                neg_score += self.negative_words[word]
        
        total = pos_score + neg_score
        return (pos_score / total) if total > 0 else 0.5


# 2. 加载数据
comments = [
    '武康路上的老洋房充满历史气息，散步很舒服，值得一来',
    '新天地的建筑很有设计感，但消费太高，有点失望',
    '豫园是上海的标志性景点，园林设计精妙，很值得看',
    '外滩看浦江夜景很美，但人太多了，拍照不方便',
    '田子坊有很多特色小店，创意十足，逛了半天还没逛够',
    '城隍庙的传统建筑保护得很好，感受到了老上海的文化',
    '陆家嘴的高楼很壮观，但缺少人文气息',
    '安福路很安静，古树很多，特别适合散步',
    '思南公馆的环境很优雅，适合拍照',
    '静安寺很宁静祥和，虽然人多但很有氛围',
]

df = pd.DataFrame({'content': comments})

# 3. 打卡点列表
landmarks = [
    '外滩', '南京路', '豫园', '城隍庙', '田子坊', '新天地',
    '武康路', '安福路', '思南公馆', '静安寺', '陆家嘴'
]

# 4. 提取打卡点评论
landmark_comments = defaultdict(list)
for _, row in df.iterrows():
    text = row['content']
    for landmark in landmarks:
        if landmark in text:
            landmark_comments[landmark].append(text)

print(f"识别到 {len(landmark_comments)} 个打卡点\n")

# 5. 情感分析
analyzer = SentimentAnalyzer()
results = []

for landmark, comments_list in landmark_comments.items():
    # 分析每条评论
    sentiments = [analyzer.analyze(comment) for comment in comments_list]
    
    # 统计
    avg_score = np.mean(sentiments)
    positive_count = sum(1 for s in sentiments if s > 0.6)
    negative_count = sum(1 for s in sentiments if s < 0.4)
    pos_rate = positive_count / len(sentiments)
    
    # 获取代表评论
    best_idx = np.argmax(sentiments)
    sample = comments_list[best_idx][:40]
    
    # 确定等级
    if avg_score >= 0.7:
        grade = '强正面'
    elif avg_score >= 0.6:
        grade = '正面'
    elif avg_score >= 0.4:
        grade = '中立'
    else:
        grade = '负面'
    
    results.append({
        '打卡点': landmark,
        '情感得分': round(avg_score, 3),
        '情感等级': grade,
        '积极评论': positive_count,
        '负面评论': negative_count,
        '积极率': f"{pos_rate:.1%}",
        '样本量': len(sentiments),
        '代表评论': sample
    })

# 6. 生成结果表
results_df = pd.DataFrame(results).sort_values('情感得分', ascending=False)

print("=" * 100)
print("📊 分析结果：")
print("=" * 100)
print(results_df.to_string(index=False))

# 7. 保存结果
results_df.to_csv('打卡点分析结果.csv', index=False, encoding='utf_8_sig')
print("\n✓ 结果已保存到 打卡点分析结果.csv")

# 8. 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('上海CityWalk打卡点情感分析报告', fontsize=18, fontweight='bold')

# 子图1: 情感得分
ax1 = axes[0, 0]
cmap = plt.cm.get_cmap('RdYlGn')
colors = [cmap(s) for s in results_df['情感得分']]
ax1.barh(results_df['打卡点'], results_df['情感得分'], color=colors)
ax1.set_title('情感得分排行', fontweight='bold')
ax1.set_xlabel('得分')

# 子图2: 评论数量
ax2 = axes[0, 1]
ax2.bar(range(len(results_df)), results_df['样本量'], color='skyblue')
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels(results_df['打卡点'], rotation=45, ha='right')
ax2.set_title('评论数量', fontweight='bold')

# 子图3: 积极率
ax3 = axes[1, 0]
pos_rates = [float(s.rstrip('%'))/100 for s in results_df['积极率']]
ax3.bar(range(len(results_df)), pos_rates, color='lightgreen')
ax3.set_xticks(range(len(results_df)))
ax3.set_xticklabels(results_df['打卡点'], rotation=45, ha='right')
ax3.set_title('积极评论率', fontweight='bold')
ax3.set_ylim(0, 1)

# 子图4: 等级分布
ax4 = axes[1, 1]
grade_counts = results_df['情感等级'].value_counts()
ax4.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.0f%%')
ax4.set_title('情感等级分布', fontweight='bold')

plt.tight_layout()
plt.savefig('打卡点分析报告.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存到 打卡点分析报告.png")

# 9. 深度洞察
print("\n" + "=" * 100)
print("💡 深度洞察分析：")
print("=" * 100)

overall_score = results_df['情感得分'].mean()
if overall_score >= 0.7:
    desc = "🌟 高度推荐"
elif overall_score >= 0.6:
    desc = "😊 值得体验"
elif overall_score >= 0.5:
    desc = "😐 一般"
else:
    desc = "😞 需谨慎"

print(f"\n📊 整体评估：")
print(f"   • 综合得分：{overall_score:.3f}/1.0 - {desc}")
print(f"   • 总样本数：{results_df['样本量'].sum()} 条评论")
print(f"   • 分析地点：{len(results_df)} 个")

print(f"\n🏆 Top 3 推荐：")
for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
    stars = "⭐" * int(row['情感得分'] * 5)
    print(f"   {i}. {row['打卡点']:10s} | 得分: {row['情感得分']:.3f} {stars}")

print("\n✨ 分析完成！")
```

---

## 🎯 如何使用你自己的数据

### 第1步：准备数据文件

创建 CSV 文件 `my_data.csv`：
```csv
content
武康路上的老洋房充满历史气息，散步很舒服，值得一来
新天地的建筑很有设计感，但消费太高，有点失望
豫园是上海的标志性景点，园林设计精妙，很值得看
```

或者 Excel 文件 `my_data.xlsx`（包含评论列）

### 第2步：修改代码加载数据

```python
# 从CSV加载
df = pd.read_csv('my_data.csv', encoding='utf-8')

# 或从Excel加载
df = pd.read_excel('my_data.xlsx')
```

### 第3步：运行分析

```python
# 按照上面的完整代码示例运行
# 程序会自动处理你的数据
```

---

## 📁 所有可用文件

| 文件 | 说明 | 何时使用 |
|------|------|---------|
| **run_analysis.py** | 主分析脚本 | 直接运行此文件 |
| **example_usage.py** | 完整使用示例 | 学习高级用法 |
| **快速启动.py** | Windows一键启动 | Windows双击运行 |
| **run.bat** | 批处理脚本 | Windows批量运行 |
| **README.md** | 完整文档（30页） | 查询详细信息 |
| **使用说明.md** | 使用指南和代码 | 学习使用方法 |
| **快速指南.txt** | 快速开始指南 | 快速上手 |
| **项目完成总结.txt** | 项目信息 | 了解项目详情 |

---

## ⚡ 核心特性总结

- ✅ **自动化分析** - 一键运行，自动完成所有步骤
- ✅ **灵活数据输入** - 支持CSV、Excel、自定义数据
- ✅ **25个打卡点** - 涵盖上海主要景点
- ✅ **中文NLP** - 60+情感词汇库，支持否定处理
- ✅ **可视化报告** - 4张专业图表和数据表格
- ✅ **可完全定制** - 修改词汇、地点、参数
- ✅ **易于集成** - 清晰的代码结构，易于二次开发

---

## 🎉 现在就开始！

```bash
# 最简单的方式
python run_analysis.py

# 或双击打开
快速启动.py
或 run.bat
```

---

**有任何问题？查看 README.md 或 使用说明.md 了解更多信息！** 🌆📊✨
