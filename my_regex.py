# -*- coding: utf-8 -*-
import re

class my_regex:
    def __init__(self):
        self.remove_special_chars = re.compile(u'[^\w\s\d\t\-\./:_áÁàÀãÃảẢạẠăĂắẮằẰẳẲặẶẵẴâÂấẤầẦẩẨậẬẫẪđĐéÉèÈẻẺẽẼẹẸêÊếẾềỀễỄểỂệỆíÍìÌỉỈĩĨịỊóÓòÒỏỎõÕọỌôÔốỐồỒổỔỗỖộỘơƠớỚờỜởỞỡỠợỢúÚùÙủỦũŨụỤưƯứỨừỪửỬữỮựỰýÝỳỲỷỶỹỸỵỴ]')
        self.remove_email = re.compile(u'\w+\d*_*@\w+[.]+\w+[.]*\w*')
        self.remove_datetime = re.compile(r'\d+[\-/]\d+[\-/]*\d*')
        self.remove_url = re.compile(u'[(https)|(http)|(ftp)|(ssh)]+://[^\s]+')
        self.remove_number = re.compile(r'\d+\w*')
        self.normalize_space = re.compile(r' +')