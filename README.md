# 基于游戏场景的LLMs(ChatGPT方向)生成类任务研究文档



### 目录：

1. 游戏文本内容相关的生成任务调研
   1. 角色扮演的竞品
   2. ChatGPT翻译优化
   3. ChatGPT用于游戏的剧情编写
   4. 使用ChatGPT来开发游戏中的角色扮演实践案例
   5. 游戏文本数据集相关收集工作
2. 非公开知识池的ChatGPT微调训练技术调研
   1. GPT模型的Fine-tuning
   2. OpenAI Cookbook指南
3. 前可获取的开源大模型(ChatGPT的替代方案)
   1. 常用开源 NLP 模型
   2. 最新的开源大模型信息--ColossalChat：完整的 ChatGPT 克隆解决方案



### 调研需求说明：

1. 游戏文本内容的文风润色、人物对话性格、NPC对话千人千面：

   - 难点：

     不同游戏文风剧情任务的切换，需要统一可复用的技术解决方案；

2. 非公开知识池的ChatGPT微调训练：

   - 验证ChatGPT的开放接口上的本地数据微调训练功能

### 

## 1、游戏文本内容相关的生成任务调研

**参考资料：**

- Paper:  《[How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)》




**相关任务：**

1. 角色扮演的竞品调研

   - 竞品：
     - [character.ai](https://beta.character.ai/)
     
       Character.AI是一个由自己的深度学习模型驱动的新产品，包括大型语言模型，从头到尾都是以对话为基础建立和训练。Character.AI认为在创造和改进端到端产品解决方案的所有部分方面会有神奇的效果。
     
       <img src=".\imgs\Figure_4.PNG" alt="Figure_4" style="zoom: 25%;" />
     
       <img src=".\imgs\Figure_7.PNG" alt="Figure_7" style="zoom:33%;" />
     
     - [inworld.ai](https://www.inworld.ai/)
     
       使用由人工智能驱动的角色创造更真实可信的世界。Inworld 提供了一个平台，用于将高级 NPC 行为和即兴对话添加到游戏和实时媒体中。使用文本到字符的提示来创建角色个性并使用 Inworld SDK 集成到体验中。
     
       <img src=".\imgs\Figure_5.PNG" alt="Figure_5" style="zoom: 25%;" />
     
       <img src=".\imgs\Figure_6.PNG" alt="Figure_6" style="zoom:25%;" />

   输入内容：游戏文本剧本、人物人设、人物之间的关系，  

   OpenAI--个人和活动定制模型的解决方案(Playground+API)：

   - 所需的数据设置：

     文本字段描述、Temperature、自动描述--Automatic description;

   - 对模型(GPT-3)进行微调

     - OpenAI Docs -- [Fine-tuning Links](https://platform.openai.com/docs/guides/fine-tuning)

2. ChatGPT翻译：使用喜欢的作家风格进行翻译

   参考案例：(Twitter--[Link](https://twitter.com/onenewbite/status/1634715151554469888))

   1. ChatGPT翻译必用技巧: 用你喜欢的作家风格翻译。最近在读一本法语学术书，直接让ChatGPT翻译的结果很枯燥。我就想，假如是我最喜欢的作家Malcolm Gladwell 用英语写的该多好啊。Malcolm的写作风格生动有趣,易读易懂,能狠狠地抓住读者的注意力。这是ChatGPT的原始翻译:[#ChatGPT](https://twitter.com/hashtag/ChatGPT?src=hashtag_click)

      <img src=".\imgs\Figure_1.jpg" alt="Figure_1" style="zoom: 33%;" />

   2. 这是我要求它使用Malcolm的风格翻译。Prompt: Translate the following into English, Malcolm Gladwell writing style。非常好读。所以，强烈建议大家用自己最熟悉的作者风格进行翻译(前提是这个作者有大量的文字在互联网上存在). P.S. 我选择翻译成英文是因为很多专业术语最终要去英文资料里找。

      <img src=".\imgs\Figure_2.jpg" alt="Figure_2" style="zoom: 33%;" />
   
3. ChatGPT用于游戏的剧情编写(回答来源于ChatGPT)

   可以通过以下步骤来实现：

   1. 收集游戏剧情中的原始文本：将游戏中的剧情文本收集并整理成适当的格式，例如每个场景或剧情片段都有一个独立的文本文件。
   2. Fine-tune ChatGPT：通过Fine-tune ChatGPT来训练一个专门用于游戏剧情扩写的模型。Fine-tune过程中，您可以将原始文本作为训练数据集，同时定义一个适当的文本生成任务，以帮助ChatGPT学习游戏剧情的风格和语言。
   3. 生成新的剧情文本：使用Fine-tuned ChatGPT生成新的游戏剧情文本，您可以指定一些初始条件，例如游戏中的当前情况，要生成的长度等，以产生与当前游戏剧情相关的新文本。
   4. 整合新的剧情文本：将生成的新文本整合到游戏中。您可以编写一个小型的游戏插件，将生成的文本与游戏当前情况结合，以生成一个新的游戏剧情。

   以下是一个用ChatGPT生成Dungeons and Dragons游戏剧情的示例：https://github.com/nshepperd/gpt-2/blob/master/playbook.ipynb

   该示例演示了如何将Dungeons and Dragons游戏的角色和场景描述提供给ChatGPT，以生成新的游戏剧情。您可以参考这个示例，并根据您的需要进行修改，以将ChatGPT应用于您的游戏剧情扩写项目。

4. 使用ChatGPT来开发游戏中的角色扮演实践案例

   使用ChatGPT来开发角色扮演游戏：

   1. 确定游戏主题和角色类型：您需要确定游戏的主题和角色类型，例如奇幻、科幻、历史等等。然后，您可以考虑游戏中的角色类型，例如勇士、法师、商人、冒险家等等。
   2. 收集角色信息和背景：为每个角色编写详细的信息和背景故事，包括角色的个性、能力、经历、目标等等。这些信息可以帮助玩家更好地了解角色，并决定自己的角色扮演方式。
   3. 定义玩家的交互方式：在游戏中，玩家需要与角色进行互动。您可以通过ChatGPT来实现角色的自然语言对话，并根据玩家的回答来决定角色的行动和反应。
   4. 进行测试和优化：一旦您开发出了ChatGPT模型，您需要对其进行测试和优化，以确保它可以很好地适应您的游戏。您可能需要通过Fine-tuning和调整模型参数来提高ChatGPT的准确性和响应速度。

   以下是一些使用ChatGPT开发角色扮演游戏的参考案例：

   1. AI Dungeon: 这是一个使用GPT-3的在线角色扮演游戏，玩家可以与角色进行自由的对话，并决定自己的角色扮演方式。该游戏使用了GPT-3来生成故事情节和角色对话。
   2. Replika: 这是一个使用人工智能的聊天机器人应用程序，玩家可以与其进行对话，并进行情感交流和个人成长。该应用程序使用了类似于ChatGPT的技术来生成自然语言对话。

   网站链接：

   1. AI Dungeon: https://play.aidungeon.io/
   2. Replika: https://replika.ai/

5. 游戏文本数据集相关收集工作

   游戏剧情和对话的相关文本数据信息(回答来源于ChatGPT)：

   寻找游戏剧情和对话的相关文本数据可以从以下几个渠道入手：

   1. 游戏开发工具包（SDK）：一些游戏开发工具包提供了许多游戏剧情和对话文本的范例，可以通过这些范例来获取文本数据。
   2. 游戏论坛和社区：游戏论坛和社区是寻找游戏剧情和对话文本数据的好去处，可以通过搜索论坛和社区获取相关的数据信息。例如，Steam 社区、Reddit 上的游戏主题论坛等。
   3. 游戏相关的网站：有一些网站专门提供游戏剧情和对话文本数据，可以通过这些网站获取大量的数据。例如，游戏数据库网站、游戏解说视频网站等。
   4. 网络爬虫：使用网络爬虫工具可以在互联网上搜索并获取游戏剧情和对话文本数据。不过需要注意的是，在进行网络爬取时要注意合法性和版权问题。
   5. 开放数据集：一些开放数据集也提供了游戏剧情和对话的文本数据，例如 Kaggle 上的游戏数据集。

   以下是几个可以获取游戏剧情和对话文本数据的网站链接和相关资源：

   1. 游戏开发工具包（SDK）：不同游戏开发平台和工具包提供的 SDK 范例不同，可以通过相应的官方文档和开发者社区获取。例如 Unity 的官方文档：https://docs.unity3d.com/Manual/ExampleScenes.html
   2. 游戏论坛和社区：在论坛和社区中，可以通过搜索特定游戏的名称或者游戏开发者的名字来获取相关信息。例如 Steam 社区：https://steamcommunity.com/
   3. 游戏相关的网站：以下是一些提供游戏数据和文本数据的网站：
      - 游戏数据库网站 Giant Bomb：https://www.giantbomb.com/
      - 游戏数据集网站 Kaggle：https://www.kaggle.com/
      - 游戏解说视频网站 Twitch：https://www.twitch.tv/
   4. 网络爬虫：可以使用 Python 爬虫框架（如 Scrapy、BeautifulSoup 等）进行网络爬取，不过需要注意合法性和版权问题。
   5. 开放数据集：以下是一些包含游戏剧情和对话文本数据的数据集：
      - Steam 游戏评论数据集：https://www.kaggle.com/najzeko/steam-game-reviews
      - Blizzard 游戏日志数据集：https://www.kaggle.com/schallerdavid/blizzard-game-logs
      - 角色扮演游戏对话数据集：https://github.com/Morizeyao/GPT2-Chinese/tree/master/data



## 2、非公开知识池的ChatGPT微调训练技术调研

### OpenAI -- GPT模型的Fine-tuning

>  **学习如何为应用定制一个模型。**

**参考资料：**

- OpenAI Documentation -- [Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- GitHub--[OpenAI Cookbook](https://github.com/openai/openai-cookbook)

### **Fine-tuning**

#### Introduction

微调可以让你从通过API提供的模型中获得更多：

1. 比提示设计(prompt design)更高质量的结果；
2. 能够在更多的例子上进行训练，而不是在提示中进行训练；
3. 由于较短的提示，可以节省tokens；
4. 更低的延迟请求；

GPT-3已经对来自开放互联网的大量文本进行了预训练。当给它只有几个例子的提示时，它往往能直觉知道你要执行什么任务，并产生一个合理的完成，这通常被成为“few-shot learning”--少量学习。

微调通过对更多的例子进行训练来改进少量的学习，使你在大量的任务中取得更好的结果。**一旦一个模型被微调，你就不需要再在提示中提供例子**。这节省了成本，并实现了更低的延迟请求。

在高层次上，微调包括以下步骤：

1. 准备并上传训练数据；
2. 训练一个新的微调模型；
3. 使用微调好的模型；



#### 哪些模型可以进行微调

微调目前只适用于以下基础模型：davinci`, `curie`, `babbage和ada。这些是原始模型，在训练后没有任何指令（例如像text-davinci-003那样）。你也能够继续微调一个微调过的模型，增加额外的数据，而不必从头开始。

#### Installation

建议使用我们的OpenAI命令行界面（CLI）。要安装它，请运行：

```shell
pip install --upgrade openai
```

设置OPENAI_API_KEY环境变量的方法是：在shell初始化脚本（如.bashrc、zshrc等）中加入以下一行，或在微调命令前的命令行中运行：

```shell
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

##### Prepare training data

训练数据是你如何教GPT-3你想让它说什么。

训练数据必须是一个JSONL文件，其中每一行是一个提示-完成对，对应一个训练实例。可以使用OpenAI的CLI数据准备工具，将训练数据转换成这种文件格式。

```txt
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```

为微调设计提示语和完成语与为OpenAI的基础模型（Davinci, Curie, Babbage, Ada）设计提示语不同。特别是，基础模型的提示通常由多个例子组成（"少量学习"），而对于微调，每个训练例子通常由一个输入例子和其相关的输出组成，不需要给出详细的说明或在同一提示中包括多个例子。

关于如何为各种任务准备训练数据的更详细指导，请参考OpenAI的[准备数据集](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)的最佳实践。

你拥有的训练实例越多，就越好。OpenAI建议至少要有几百个例子。一般来说，OpenAI发现数据集的规模每增加一倍，模型的质量就会直线上升。

##### CLI data preparation tool

OpenAI开发了一个工具，可以验证、提供建议并重新格式化你的数据：

```shell
openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
```

这个工具接受不同的格式，唯一的要求是它们包含一个提示和一个完成列/键。你可以传递一个CSV、TSV、XLSX、JSON或JSONL文件，在指导你完成建议的修改过程后，它将把输出保存到JSONL文件中，准备进行微调。

##### Create a fine-tuned model

使用OpenAI CLI开始你的微调工作：

```shell
openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>
```

其中BASE_MODEL是你的基础模型的名称（ada、babbage、curie或davinci）。你可以使用后缀参数来定制你的微调模型的名称。

运行上述命令可以做几件事：

1. 使用files API上传文件（或使用一个已经上传的文件）
2. 创建一个微调工作
3. 流动事件，直到作业完成（这通常需要几分钟，但如果队列中有许多作业或你的数据集很大，可能需要几个小时）

每个微调工作都从一个基本模型开始，默认为Curie-居里。模型的选择影响着模型的性能和运行微调模型的成本。你的模型可以是：Ada、Babbage、Curie或Davinci之一。请访问定价页面，了解微调费用的细节。

在你开始一个微调作业后，可能需要一些时间来完成。你的工作可能排在我们系统中其他工作的后面，根据模型和数据集的大小，训练我们的模型可能需要几分钟或几小时。如果事件流因任何原因被中断，你可以通过运行来恢复它：

```shell
openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>
```

当工作完成后，它应该显示微调后的模型的名称。除了创建微调作业外，你还可以列出现有的作业，检索作业的状态，或取消一个作业。

```shell
# List all created fine-tunes
openai api fine_tunes.list

# Retrieve the state of a fine-tune. The resulting object includes
# job status (which can be one of pending, running, succeeded, or failed)
# and other information
openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>

# Cancel a job
openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>
```

#### Use a fine-tuned model

当一个作业成功后，fine_tuned_model字段将被填充为模型的名称。你现在可以指定这个模型作为我们的完成度API的参数，并使用Playground向它发出请求。

在你的工作首次完成后，你的模型可能需要几分钟的时间来准备处理请求。如果对你的模型的完成请求超时，这可能是因为你的模型仍在加载中。如果发生这种情况，过几分钟再试。

你可以通过传递模型名称作为完成请求的模型参数开始提出请求：

```shell
# OpenAI CLI:
openai api completions.create -m <FINE_TUNED_MODEL> -p <YOUR_PROMPT>

# cURL:
curl https://api.openai.com/v1/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": YOUR_PROMPT, "model": FINE_TUNED_MODEL}'
```

```python
# Python
import openai
openai.Completion.create(
    model=FINE_TUNED_MODEL,
    prompt=YOUR_PROMPT)
```

```js
# Node.js
const response = await openai.createCompletion({
  model: FINE_TUNED_MODEL
  prompt: YOUR_PROMPT,
});
```

你可以继续在这些请求上使用所有其他的完成度参数，如 `temperature`, `frequency_penalty`, `presence_penalty`等，以微调模型。

#### **Preparing your dataset**

微调是一种强大的技术，可以创建一个针对你的用例的新模型。在微调你的模型之前，我们强烈建议你阅读下面这些最佳实践和针对你的用例的具体指南。

**ChatGPT进行Fine-tune大概需要准备的数据量：**

Fine-tune的数据量要求会因不同的任务和应用场景而有所不同，但是通常来说，Fine-tune ChatGPT所需的数据量要比从头开始训练模型所需的数据量要少得多。一般来说，Fine-tune ChatGPT所需的数据量应该在几千到几十万之间。

具体而言，Fine-tune ChatGPT的数据量取决于以下因素：

1. 任务类型：不同类型的任务需要的数据量是不同的。例如，生成式对话模型需要更多的对话数据，而文本分类模型可能只需要几千个标记的样本数据。
2. 数据质量：数据质量对模型的影响非常大。如果您使用的数据质量很高，那么您可能只需要很少的数据量就可以获得良好的模型表现。相反，如果您使用的数据质量很低，那么您需要更多的数据量来训练模型。
3. 模型大小和参数数量：模型的大小和参数数量也会影响所需的数据量。更大的模型需要更多的数据来Fine-tune。

总的来说，如果想Fine-tune ChatGPT，建议首先尝试使用较小的数据集进行Fine-tune，以获得初步的结果。如果觉得模型的性能可以进一步提高，那么可以尝试使用更大的数据集来Fine-tune模型。

**Data formatting**

为了微调一个模型，你需要一组训练实例，每个实例由一个输入（"提示"）和其相关的输出（"完成"）组成。这与使用我们的基础模型明显不同，在那里你可能会在一个提示中输入详细的指令或多个例子。

- 每个提示应该以一个固定的分隔符结束，以告知模型提示何时结束，完成何时开始。一般来说，一个简单的分隔符效果很好，就是`\n\n###\n\n`。分隔符不应该出现在任何提示中的其他地方。
- 由于OpenAI的标记化，每个完成度都应该以一个空白开始，因为OpenAI的标记化是以前面的空白来标记大多数的词。
- 每个完成度都应该以一个固定的停止序列结束，以告知模型该完成度何时结束。停止序列可以是 `\n`, `###`, 或任何其他不在任何完成中出现的标记。
- 对于推理，你应该以创建训练数据集时的相同方式来格式化你的提示，包括相同的分隔符。同时指定相同的停止序列以正确地截断完成。

### 一般的最佳实践--General best practices

微调在更多高质量的例子中表现得更好。为了微调一个模型，**比使用高质量的提示与我们的基础模型表现更好，你应该提供至少几百个高质量的例子，最好是由人类专家审核的**。从这里开始，性能往往随着例子数量的每增加一倍而线性增加。**增加例子的数量通常是提高性能的最好和最可靠的方法**。

分类器是最容易上手的模型。对于分类问题，我们建议使用ada，一旦经过微调，它的性能通常只比更有能力的模型差一点，同时速度明显更快，成本更低。

如果你是在预先存在的数据集上进行微调，而不是从头开始写提示语，那么，如果可能的话，一定要手动审查你的数据，看看是否有冒犯性或不准确的内容，如果数据集很大，则要尽可能多地审查随机样本。

#### 具体准则--Specific guidelines

**有条件的生成--Conditional generation**

> 条件性生成是一个需要在给定某种输入的情况下生成内容的问题。这包括转述、总结、实体提取、给定规格的产品描述写作、聊天机器人和许多其他问题。对于这种类型的问题，OpenAI建议：
>
> - 在提示的最后使用一个分隔符，例如：`\n\n###\n\n`。当你最终向你的模型提出请求时，记得也要附加这个分隔符。
> - 在完成度的最后使用一个结束语，例如 `END`
> - 在推理过程中，记得将结束标记作为停止序列加入，例如，`stop=[" END"]`
> - 争取至少有~500个例子
> - 确保提示+完成度不超过2048个令牌，包括分隔符
> - 确保实例的质量和遵循相同的预期格式
> - 确保用于微调的数据集在结构和任务类型上与模型将要使用的非常相似。
> - 对于这些用例，使用较低的学习率和只有1-2个epochs往往效果更好。

**Case study 1: Write an engaging ad based on a Wikipedia article**

- 案例研究：根据维基百科上的文章写一个吸引人的广告

这是一个生成性用例，所以你要确保你提供的样本是最高质量的，因为微调后的模型将试图模仿所给的例子的风格（和错误）。一个好的起点是500个左右的例子。一个样本数据集可能看起来像这样：

```json
{"prompt":"<Product Name>\n<Wikipedia description>\n\n###\n\n", "completion":" <engaging ad> END"}
```

For example:

```json
{"prompt":"Samsung Galaxy Feel\nThe Samsung Galaxy Feel is an Android smartphone developed by Samsung Electronics exclusively for the Japanese market. The phone was released in June 2017 and was sold by NTT Docomo. It runs on Android 7.0 (Nougat), has a 4.7 inch display, and a 3000 mAh battery.\nSoftware\nSamsung Galaxy Feel runs on Android 7.0 (Nougat), but can be later updated to Android 8.0 (Oreo).\nHardware\nSamsung Galaxy Feel has a 4.7 inch Super AMOLED HD display, 16 MP back facing and 5 MP front facing cameras. It has a 3000 mAh battery, a 1.6 GHz Octa-Core ARM Cortex-A53 CPU, and an ARM Mali-T830 MP1 700 MHz GPU. It comes with 32GB of internal storage, expandable to 256GB via microSD. Aside from its software and hardware specifications, Samsung also introduced a unique a hole in the phone's shell to accommodate the Japanese perceived penchant for personalizing their mobile phones. The Galaxy Feel's battery was also touted as a major selling point since the market favors handsets with longer battery life. The device is also waterproof and supports 1seg digital broadcasts using an antenna that is sold separately.\n\n###\n\n", "completion":"Looking for a smartphone that can do it all? Look no further than Samsung Galaxy Feel! With a slim and sleek design, our latest smartphone features high-quality picture and video capabilities, as well as an award winning battery life. END"}
```

这里我们使用了一个多行分隔符，因为维基百科的文章包含多个段落和标题。我们还使用了一个简单的结束符，以确保模型知道完成的时间。

**Case study 2: Entity extraction**

- 案例研究：实体提取

这类似于一个语言转换任务。为了提高性能，最好是将不同的提取的实体按字母顺序或按它们在原文中出现的顺序进行排序。这将有助于模型按顺序跟踪所有需要生成的实体。该数据集可以如下：

```json
{"prompt":"<any text, for example news article>\n\n###\n\n", "completion":" <list of entities, separated by a newline> END"}
```

For ezample:

```json
{"prompt":"Portugal will be removed from the UK's green travel list from Tuesday, amid rising coronavirus cases and concern over a \"Nepal mutation of the so-called Indian variant\". It will join the amber list, meaning holidaymakers should not visit and returnees must isolate for 10 days...\n\n###\n\n", "completion":" Portugal\nUK\nNepal mutation\nIndian variant END"}
```

多行分隔符效果最好，因为文本可能包含多行。理想情况下，输入提示的类型会有很高的多样性（新闻文章、维基百科网页、推特、法律文件），这反映了提取实体时可能遇到的文本。

**Case study 3: Customer support chatbot**

- 案例研究：客服聊天机器人

聊天机器人通常会包含关于对话的相关背景（订单细节），到目前为止的对话摘要以及最近的信息。对于这个用例，同一个过去的对话可以在数据集中产生多行，每次都有稍微不同的上下文，对于每个代理的生成都是一种完成。**这个用例需要几千个例子，因为它可能会处理不同类型的请求，以及客户问题**。为了确保性能的高质量，我们建议对对话样本进行审核，以确保代理信息的质量。摘要可以用一个单独的文本转换微调模型来生成。数据集可以看起来如下：

```json
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent:", "completion":" <response2>\n"}
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent: <response2>\nCustomer: <message3>\nAgent:", "completion":" <response3>\n"}
```

在这里，我们特意将不同类型的输入信息分开，但在提示和完成之间保持客户代理对话框的格式不变。所有的完成都应该只由代理来完成，在做推理时我们可以使用 `\n` 作为停止序列。



### Advanced usage--高阶用法

#### 定制你的模型名称

你可以使用后缀参数为你的微调模型名称添加一个最多 40 个字符的后缀。

OpenAI CLI：

```shell
openai api fine_tunes.create -t test.jsonl -m ada --suffix "custom model name"
```

由此产生的名称将是：

```txt
ada:ft-your-org:custom-model-name-2022-02-15-04-21-04
```

#### 分析微调模型

每项工作完成后，OpenAI都会给它附加一个结果文件。当你检索微调时，这个结果文件的ID将被列出，当你查看微调的事件时也是如此。你可以下载这些文件：

OpenAI CLI:

```shell
openai api fine_tunes.results -i <YOUR_FINE_TUNE_JOB_ID>
```

CURL:

```shell
curl https://api.openai.com/v1/files/$RESULTS_FILE_ID/content \
  -H "Authorization: Bearer $OPENAI_API_KEY" > results.csv
```

`_results.csv` 文件包含每一个训练步骤的一行，其中一个步骤指的是对一批数据的前向和后向传递。除了步骤编号外，每一行还包含与该步骤相对应的下列字段：

- **elapsed_tokens**: 

  到目前为止，该模型所看到的tokens数量（包括重复的）。

- **elapsed_examples**: 

  模型到目前为止看到的例子的数量（包括重复的），其中一个例子是你的批次中的一个元素。例如，如果 batch_size = 4，每一步将增加 elapsed_examples 4。

- **training_loss**: 

  训练批次的损失

- **training_sequence_accuracy**: 

  在训练批次中，模型预测的标记与真实完成的标记完全吻合的百分比。例如，在 batch_size 为 3 的情况下，如果你的数据包含完成度 [[1, 2], [0, 5], [4, 2]] 而模型预测 [[1, 1], [0, 5], [4, 2]] ，这个准确性将是 2/3 = 0.67

- **training_token_accuracy**: 

  训练批次中被模型正确预测的标记的百分比。例如，在 batch_size 为 3 的情况下，如果你的数据包含完成 [[1, 2], [0, 5], [4, 2]] 而模型预测 [[1, 1], [0, 5], [4, 2]] ，这个准确性将是 5/6 = 0.83



#### Hyperparameters--超参数

OpenAI已经挑选了默认的超参数，这些参数在一系列的使用情况下都能很好地工作。唯一需要的参数是训练文件。

也就是说，调整用于微调的超参数往往可以使模型产生更高质量的输出。特别是，你可能想配置以下内容：

- `model`

  要微调的基本模型的名称。你可以选择 "ada"、"babbage"、"curie "或 "davinci "中的一个。要了解这些模型的更多信息，请看[模型文档](https://platform.openai.com/docs/models)。

- `n_epochs`

  默认为4。 训练模型的历时数。一个epoch是指训练数据集的一个完整周期。

- `batch_size`

  默认为训练集中实例数量的~0.2%，上限为256。批量大小是指用于训练单个前向和后向的训练例子的数量。一般来说，我们发现较大的批次大小往往对较大的数据集有更好的效果。

- `learning_rate_multiplier`

  默认为0.05、0.1或0.2，取决于最终的批次大小。微调学习率是用于预训练的原始学习率乘以这个乘数。OpenAI建议在0.02到0.2的范围内进行试验，看看什么能产生最好的结果。根据经验，OpenAI发现较大的学习率往往在较大的批次规模下表现更好。

- `compute_classification_metrics`

  默认为 "假"。如果是 "真"，为了对分类任务进行微调，在每个历时结束时对验证集计算特定的分类指标（准确度、F-1得分等）。

要配置这些额外的超参数，可以通过OpenAI CLI上的命令行标志传入，比如说：

```shell
openai api fine_tunes.create \
  -t file-JD89ePi5KMsB3Tayeli5ovfW \
  -m ada \
  --n_epochs 1
```

**Continue fine-tuning from a fine-tuned model**

如果你已经为你的任务微调了一个模型，现在有了你想纳入的额外训练数据，你可以继续从模型中微调。这将创建一个从所有训练数据中学习的模型，而不必从头开始重新训练。

要做到这一点，在创建一个新的微调任务时，传入微调模型的名称（例如，-m curie:ft-<org>-<date>）。其他训练参数不必改变，但是如果你的新训练数据比以前的训练数据小得多，你可能会发现将学习率乘数减少2到4倍是很有用的。



### **OpenAI Cookbook**

The OpenAI Cookbook分享了使用OpenAI API完成常见任务的示例代码。要运行这些例子，你需要一个OpenAI账户和相关的API密钥。

### 指南和示例

- API 使用方法
  - [如何处理速度限制](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb)
    - [避免触及速率限制的并行处理脚本示例](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)
  - [如何用tiktoken计算tokens](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
  - [如何流转完成度](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)
- ChatGPT
  - [如何格式化ChatGPT模型的输入](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)
- GPT-3
  - [指南：如何与大型语言模型一起工作](https://github.com/openai/openai-cookbook/blob/main/how_to_work_with_large_language_models.md)
  - [指南：提高可靠性的技术](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)
  - [如何使用多步骤提示来编写单元测试](https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt.ipynb)
  - [文本写作实例](https://github.com/openai/openai-cookbook/blob/main/text_writing_examples.md)
  - [代码编写实例](https://github.com/openai/openai-cookbook/blob/main/code_writing_examples.md)
  - [代码解释示例](https://github.com/openai/openai-cookbook/blob/main/code_explanation_examples.md)
- Embeddings
  - [文本比较实例](https://github.com/openai/openai-cookbook/blob/main/text_comparison_examples.md)
  - [如何获得embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Get_embeddings.ipynb)
  - [使用嵌入技术回答问题](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
  - [使用嵌入的语义搜索](https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb)
  - [使用嵌入技术的推荐](https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb)
  - [聚类嵌入](https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb)
  - 在[二维](https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_2D.ipynb)或[三维](https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_3D.ipynb)中实现嵌入的可视化
  - [嵌入长文本](https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb)
- Fine-tuning GPT-3
  - [指南：对GPT-3进行微调以对文本进行分类的最佳做法](https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit)
  - [微调后的分类](https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb)
- DALL-E
  - [如何用DALL-E生成和编辑图像](https://github.com/openai/openai-cookbook/blob/main/examples/dalle/Image_generations_edits_and_variations_with_DALL-E.ipynb)

### 近期的新增内容：

- [如何格式化ChatGPT模型的输入](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)

- [使用Redis的向量数据库进行嵌入搜索](https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases/redis)

- [网站问答与嵌入](https://github.com/openai/openai-cookbook/tree/main/apps/web-crawl-q-and-a)

- [带嵌入的文件问答](https://github.com/openai/openai-cookbook/tree/main/apps/file-q-and-a)

  文件问答是一款[Next.js](https://nextjs.org/)应用程序，可让您使用 OpenAI API 在文件中查找答案。您可以上传文件并提出与其内容相关的问题，该应用程序将使用嵌入和 GPT 从最相关的文件中生成答案。

  原理解析：上传文件时，将从文件中提取文本。然后将此文本拆分为较短的文本块，并为每个文本块创建一个嵌入。当用户提出问题时，会为该问题创建嵌入，并执行相似性搜索以找到与问题最相似的文件块嵌入（即与问题嵌入具有最高余弦相似性）。然后对完成端点进行 API 调用，问题和最相关的文件块包含在提示中。如果可以在提取物中找到答案，则生成模型会给出在文件块中找到的问题的答案。

- 使用[Weights & Biases](https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_W%26B.ipynb)可视化嵌入

- [使用Pinecone的检索增强生成性问题回答](https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/pinecone/Gen_QA.ipynb)



### GPT-3微调的其他示例：

- GPT-3文本生成: https://github.com/imcaspar/gpt3/tree/main/examples
- Hugging Face: https://github.com/huggingface



## 3、目前可获取的开源大模型(ChatGPT的替代方案)

### 1、常用开源 NLP 模型的入门案例和相关链接：

1. BERT 模型：

- BERT Fine-tuning Tutorial with PyTorch：https://mccormickml.com/2019/07/22/BERT-fine-tuning/
- Fine-tuning a pre-trained BERT model for sequence classification with PyTorch：https://towardsdatascience.com/fine-tuning-a-pre-trained-bert-model-for-sequence-classification-with-pytorch-382fdd047828

2. GPT-2 模型：

- GPT-2 Text Generation with Python and OpenAI’s GPT-2：https://towardsdatascience.com/gpt-2-text-generation-with-python-and-openais-gpt-2-6a64b38f8b58
- GPT-2 for Text Generation with PyTorch：https://huggingface.co/blog/how-to-generate

3. RoBERTa 模型：

- A step-by-step guide to RoBERTa: Preprocessing, training, and decoding：https://towardsdatascience.com/a-step-by-step-guide-to-roberta-preprocessing-training-and-decoding-eeaa83305a3d
- Fine-tuning RoBERTa for Sentiment Analysis：https://colab.research.google.com/drive/1G0wYcz-ImE6uAKJhUgKjzU6cwVKy6LJ2?usp=sharing

4. T5 模型：

- How to Use T5 Model for Text Classification：https://towardsdatascience.com/how-to-use-t5-model-for-text-classification-3d6896445f5f
- Pretraining and Fine-tuning T5 transformer for text-to-SQL generation：https://towardsdatascience.com/pretraining-and-fine-tuning-t5-transformer-for-text-to-sql-generation-206d3cb7113d

**这些模型进行训练需要的算力：**

对于训练大型NLP模型，所需的算力和时间通常取决于以下几个因素：

1. 模型的大小和复杂度：较大和更复杂的模型需要更多的算力和时间来训练。
2. 训练数据的规模：更大的训练数据集需要更多的算力和时间来训练模型。
3. 训练时的超参数设置：超参数设置如学习率、批量大小、训练轮数等都会影响训练的速度和效果。

在实际应用中，需要根据具体情况来确定所需的算力和时间。一般来说，对于大型NLP模型的训练，需要使用高性能计算机（如GPU或TPU），以加快训练速度和提高效率。通常来说，如果只是针对某个具体的任务 Fine-tune 预训练模型，使用单个GPU就可以，但是如果需要从头开始训练大型模型，则需要更高的计算资源。

以目前常见的几个大型NLP模型为例，一些常见的训练时间和所需的算力如下：

- BERT：对于一些小型任务，使用单个GPU，训练时间可能在几小时到一天之间。对于更大的任务，可能需要使用多个GPU或TPU，训练时间可能需要几天到几周。
- GPT-2：使用单个GPU训练 GPT-2 可能需要数天到数周，使用多个GPU或TPU可显著加快训练速度。
- RoBERTa：使用单个GPU训练 RoBERTa 可能需要数天到数周，对于更大的模型，可能需要使用多个GPU或TPU。
- T5：T5 模型比其他模型要复杂，使用单个GPU训练可能需要几天到几周，使用多个GPU或TPU可能需要更长时间。

### <span style='color:brown'>**2、最新的开源大模型信息**</span>

- **ColossalChat--完整的 ChatGPT 克隆解决方案**

  - GitHub：
    - https://github.com/hpcaitech/ColossalAI
  - [ColossalChat：使用完整的 RLHF 管道克隆 ChatGPT 的开源解决方案](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)

  第一个完整的端到端模型流水线刚刚发布，是目前最实用的类似于ChatGPT的开源项目。

  Colossal-AI 发布了基于 LLaMA 预训练模型的开源 RLHF 流水线，包括： • 监督数据收集 • 监督微调 • 奖励模型训练 • 强化学习微调 他们称之为“ColossalChat”。

  这是最令人印象深刻的部分： ColossalChat 需要不到 100 亿个参数就能在英文和中文中获得与 ChatGPT 和 GPT-3.5 相当的结果。

  >  ColossalAI是一个旨在使大型人工智能模型更加便宜、更快、更易于访问的项目。该项目提供了一系列并行组件，支持用户像在本地计算机上编写模型一样进行分布式深度学习模型训练和推理。它支持多种并行策略，包括数据并行、管道并行、张量并行、序列并行、零冗余优化、自动并行等。此外，该项目还提供了多个示例，并且已经应用到了实际的应用程序中，例如生成式对话系统和蛋白质结构预测。