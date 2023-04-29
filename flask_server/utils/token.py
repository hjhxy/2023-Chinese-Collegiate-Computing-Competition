from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from itsdangerous import BadSignature, SignatureExpired

SECRET_KEY = 'zxyyyds__123456'  # 这里可以设置密钥


class Token:

    # 生成token，有效时间为24h
    @staticmethod
    def generate_auth_token(username, password, expiration=3600 * 24):
        s = Serializer(SECRET_KEY, expires_in=expiration)
        return s.dumps({'username': username, 'password': password}).decode()

    # 解析token
    @staticmethod
    def verify_auth_token(token):
        s = Serializer(SECRET_KEY)
        try:
            # token正确
            data = s.loads(token)
            return data
        except SignatureExpired:
            # token过期
            print("token已经过期")
            return None
        except BadSignature:
            # token错误
            print("token错误")
            return None
