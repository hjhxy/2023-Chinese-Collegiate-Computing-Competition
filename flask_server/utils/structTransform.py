class objDictTool:
    @staticmethod
    def to_dic(obj):
        dic = {}
        for fieldkey in dir(obj):
            fieldvaule = getattr(obj, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvaule) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvaule
        return dic




