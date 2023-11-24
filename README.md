# 2023年中国大学生计算机设计大赛
*《基于改进YOLOv7与无人机辅助的城市道路质量监测系统》* 
*人工智能赛道省一等奖 国家二等奖*  

2023013247-项目源码与数据集

## 项目说明
|--flask_server：后端代码
|--vue-admin-template：前端代码
|--yolo_ssd：目标检测代码与模型权重

### 2.1 拉取项目 
git clone https://github.com/hjhxy/2023-Chinese-Collegiate-Computing-Competition.git

### 2.2 项目运行
注意：在项目运行之前确保克隆的项目都在一个文件夹下，请勿随意更换三个子模块的相对位置<br/>
|--flask_server：后端代码<br/>
    在项目根路径打开终端cmd<br/>
    编译工具：pycharm/vscode<br/>
    依赖下载：pip install -r requirePackage.txt <br/>
    运行：set FLASK_APP=main.py ; <br/>
          flask run<br/>
|--vue-admin-template：前端代码<br/>
    在项目根路径打开终端cmd<br/>
    编译工具：webstorm/vscode<br/>
    依赖下载：npm i <br/>
    运行：npm run dev 即可在浏览器看到项目运行结果(如未自动打开，请点击命令行的链接或浏览器输入http://localhost:9528访问)<br/>
|--yolo_ssd：pytorch目标检测代码:
    该检测模型的运行基于上述项目文件调用，无需额外配置
    但需要提前配置好相应包文件，见其中的requirement.txt
    
    
    
