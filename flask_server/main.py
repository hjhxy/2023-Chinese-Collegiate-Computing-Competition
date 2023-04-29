# 这是一个示例 Python 脚本。
import os.path
import random

from flask import Flask, request, send_from_directory, jsonify
from gevent import pywsgi
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, UniqueConstraint
from flask_migrate import Migrate
from utils.token import Token
from utils.getTime import getTime
from utils.structTransform import objDictTool
import sys

sys.path.append("..")
from yolo_ssd.yolo_predict import predict_img, predict_video

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:zxy3057472776@127.0.0.1:3306/jishe'
app.config['SQLALCHEMY_ECHO'] = True
CORS(app, resource={r"/*": {"origins": "*"}})

db = SQLAlchemy(app)
migrate = Migrate(app, db)
basedir = os.path.abspath(os.path.dirname(__file__))


#  user表
class User(db.Model):
    __tablename__ = "user"  # 映射得到的表名
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 代表一个列
    nickname = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(100), nullable=False)
    avatar = db.Column(db.String(300), nullable=False)
    introduction = db.Column(db.String(500), nullable=False)
    isopend = db.Column(db.Integer, nullable=False)  # 是否启用，0表示不启用，1表示启用
    permission = db.Column(db.Integer, nullable=False)  # 权限，0表示超级管理员，1表示管理员，2表示普通用户
    createtime = db.Column(db.String(20), nullable=False)  # 检测时间
    UniqueConstraint("username")


# 消息表
class Message(db.Model):
    __tablename__ = "message"  # 映射得到的表名
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 代表一个列
    request_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    msg_type = db.Column(db.String(100), nullable=False)
    msg_content = db.Column(db.String(10000), nullable=False)
    msg_status = db.Column(db.String(20), nullable=False)
    request_time = db.Column(db.String(50), nullable=False)
    handle_time = db.Column(db.String(50))
    # 直接做一个表的映射操作
    username = db.relationship("User")
    # 反映射，user表添加messages属性，可以获取对应外键id的所有记录
    author = db.relationship("User", backref="messages")


# 检测历史
class CheckHistory(db.Model):
    __tablename__ = "checkhistory"  # 映射得到的表名
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 代表一个列
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # 用户id
    check_type = db.Column(db.String(100), nullable=False)  # 检测类型 img/video
    srcurl = db.Column(db.String(1000), nullable=False)  # 原地址
    url = db.Column(db.String(1000), nullable=False)  # 地址
    check_time = db.Column(db.String(20), nullable=False)  # 检测时间
    detail = db.Column(db.String(3000), nullable=False)  # 存储图片的具体信息，拍摄时间，经纬度等
    # 直接做一个表的映射操作
    username = db.relationship("User")


# 创建所有表
with app.app_context():
    db.create_all()

token = None
username = None
password = None
black_channel_token = ['/user/resetpwd', "/uploadimg", "/uploadvideo", '/user/logout', '/user/getuserinfo',
                       '/system/userlist', '/system/msglist']


# 拦截器
@app.before_request
def before():
    url = request.path
    # print(green_channel_token.count(url))
    if not black_channel_token.count(url):
        pass
    else:
        global token, username, password
        token = request.headers.get('token')
        print(token)
        if token is None:
            return "当前未登录或长期未登录，请重新登录"
        else:
            user = Token.verify_auth_token(token)
            username = user['username']
            password = user['password']
            pass


@app.route('/', methods=['GET', 'POST'])
def index():
    # 1. 添加一条数据并提交到数据库
    user = User(nickname="admin", username="admin", password="123456", phone="15581307092",
                avatar="https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif",
                introduction="我是超级管理员", isopend="1", permission="0")
    db.session.add(user)  #
    db.session.commit()  # 同步到数据库
    # 2. get查找，根据主键查找
    # user = db.session.get(User, 1)
    # print(user.id, user.nickname, user.username)
    # 3. filter_by，查找多条 QuerySet：类数组，可以做切片等操作
    # user = User.query.filter_by(username="admin")
    # 4. 查找所有数据 User.query.all
    # 5. 查找第一条数据 user.query.first
    # 6. 更新操作,都是操作对象
    # user.nickname = '法外狂徒'
    # db.session.commit()
    # 7. 删除操作
    # db.session.delete(user)
    # db.session.commit()
    # msg = Message(request_user_id='1', msg_type='text', msg_content='文本', msg_status='0', request_time='1956565656', handle_time='')
    # db.session.add(msg)
    # db.session.commit()
    # users = User.query.all()
    # print(users[0])
    return "hello flask"


@app.route('/uploadimg', methods=['POST'])
def uploadimg():
    img = request.files['file']
    randomnum = random.randint(pow(10, 8) - 1, pow(10, 10) - 1)
    path = basedir + "\\static\\img\\"
    filename = str(randomnum) + img.filename
    img.save(path + "\\" + filename)

    img_property = predict_img(path, filename)
    srcurl = "/static/img/" + filename
    url = "/static/resImg/" + filename

    # 查找用户信息
    user = User.query.filter_by(username=username)
    if len(list(user)) != 0:
        # 添加记录到数据库
        check = CheckHistory(user_id=user[0].id, check_type="img", srcurl=srcurl, url=url, check_time=getTime(),
                             detail=jsonify(img_property))
        db.session.add(check)  #
        db.session.commit()  # 同步到数据库
        return jsonify(
            {
                "propties": img_property,
                "url": url
            }
        )
    else:
        return "用户" + username + "不存在"


@app.route('/uploadvideo', methods=['POST'])
def uploadvideo():
    video = request.files['file']
    randomnum = random.randint(pow(10, 8) - 1, pow(10, 10) - 1)
    path = basedir + "\\static\\video\\"
    filename = str(randomnum) + video.filename
    video.save(path + "\\" + filename)

    result_video = predict_video(path, filename)
    srcurl = "/static/video/" + filename
    url = "/static/resVideo/" + filename

    # 查找用户信息
    user = User.query.filter_by(username=username)
    if len(list(user)) != 0:
        # 添加记录到数据库
        check = CheckHistory(user_id=user[0].id, check_type="video", srcurl=srcurl, url=url, check_time=getTime(),
                             detail=jsonify(result_video))
        db.session.add(check)  #
        db.session.commit()  # 同步到数据库
        return jsonify({
            "result": result_video,
            "url": url
        })
    else:
        return "用户" + username + "不存在"


@app.route('/uploadcamera', methods=['POST'])
def uploadcamera():
    return 'uploadcamera'


@app.route('/user/login', methods=['POST'])
def login():
    global username, password, token
    jsondata = request.json
    username = jsondata['username']
    password = jsondata['password']
    user = User.query.filter_by(username=username)
    if len(list(user)) == 0:
        res = {
            "message": "登陆失败，账号或密码错误",
            "code": -1
        }
    else:
        token = Token.generate_auth_token(username, password)
        res = {
            "message": "login success",
            "data": {
                "token": token,
            },
            "code": 200
        }
    return jsonify(res)


# 注册/添加用户
@app.route('/user/register', methods=['POST'])
def register():
    global username, password
    jsondata = request.json
    username = jsondata['username']
    password = jsondata['password']
    user = User.query.filter_by(username=username)
    if len(list(user)) == 0:
        user = User(nickname="user", username=username, password=password, phone="1",
                    avatar="https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif",
                    introduction="我是管理员", isopend="1", permission="1", createtime=getTime())
        db.session.add(user)  #
        db.session.commit()  # 同步到数据库
        res = {
            "message": "注册成功",
            "code": 200
        }
    else:
        res = {
            "message": "注册失败，用户已存在",
            "code": -1
        }
    return jsonify(res)


@app.route('/user/resetpwd', methods=['POST'])
def forgetPwd():
    global username, password
    jsondata = request.json
    username = jsondata['username']
    password = jsondata['password']
    newpassword = jsondata['newpassword']
    user = User.query.filter_by(username=username)
    if len(list(user)) != 0 and password == user[0].password:
        user[0].password = newpassword
        db.session.commit()
        res = {
            "message": "修改成功",
            "code": 200
        }
    else:
        res = {
            "message": "用户名或密码错误，修改失败",
            "code": -1
        }
    return jsonify(res)


@app.route('/user/logout', methods=['POST'])
def logout():
    print(username, password)
    res = {
        "message": "退出登录成功",
        "code": 200,
    }
    return jsonify(res)


@app.route('/user/deleteuser', methods=['POST'])
def deleteuser():
    jsondata = request.json
    delusername = jsondata['username']
    user = User.query.filter_by(username=delusername)
    if len(list(user)) == 0:
        res = {},
    else:
        db.session.delete(user)
        db.session.commit()
        res = {
            "message": "delete success",
            "code": 200,
            "data": {},
        }
    return jsonify(res)


@app.route('/user/getuserinfo', methods=['GET'])
def getUserInfo():
    user = User.query.filter_by(username=username)
    if len(list(user)) == 0:
        res = {
            "message": "获取信息失败",
            "code": -1,
            "data": {
                "roles": ['admin'],
                "introduction": 'I am a super administrator',
                "avatar": 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
                "name": 'Super Admin'
            },
        }
    else:
        res = {
            "message": "login success",
            "code": 200,
            "data": {
                "roles": ['admin'],
                "introduction": 'I am a super administrator',
                "avatar": 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
                "name": 'Super Admin'
            },
        }
    return jsonify(res)


# 获取所有用户
@app.route('/system/userlist', methods=['GET'])
def userList():
    checklist = User.query.all()
    resultdata = []
    for item in checklist:
        dictret = dict(item.__dict__)
        dictret.pop('_sa_instance_state', None)
        resultdata.append(dictret)
    res = {
        "message": "所有用户列表",
        "data": {
            "items": resultdata
        },
        "code": 200,
    }
    return jsonify(res)


# 获取检测历史
@app.route('/system/checklist', methods=['GET'])
def checkList():
    checklist = CheckHistory.query.all()
    resultdata = []
    for item in checklist:
        dictret = dict(item.__dict__)
        dictret.pop('_sa_instance_state', None)
        resultdata.append(dictret)
    res = {
        "message": "所有检测历史",
        "data": {
            "items": resultdata
        },
        "code": 200,
    }
    return jsonify(res)


# 获取消息列表
@app.route('/system/msglist', methods=['GET'])
def msgList():
    res = {
        "message": "所有信息列表",
        "code": 200,
        "data": {
            "items": [
                {
                    "nickname": "zs",
                    "msg_type": "文本",
                    "msg_content": "你好",
                    "msg_status": "待处理",
                    "request_time": "2022-12-12",
                    "handle_time": "2022-12-25"
                }
            ]
        },
    }
    return jsonify(res)


server = pywsgi.WSGIServer(('0.0.0.0', 3001), app)  # 服务器的地址及端口号
server.serve_forever()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
