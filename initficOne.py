class Student_Grade:#与java类似,面向对象编程，构造了一个对象
    def __init__(self):  # 类似于c++中的默认构造函数
        self.name = None
        self.grade = None
    def print_grade(self):
        print("%s grade is %s" % (self.name, self.grade))


s1 = Student_Grade()  # 创建对象s1
s1.name = "Tom"
s1.grade = 8

s2 = Student_Grade()  # 创建对象s2
s2.name = "Jerry"
s2.grade = 7


s1.print_grade()
s2.print_grade()