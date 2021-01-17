class Student_Grade: #有参构造
    def __init__(self, name, grade):  #类似于C++中的有参构造函数
        self.name = name
        self.grade = grade

    def print_grade(self):
        print("%s grade is %s" % (self.name,self.grade))

s1 = Student_Grade("Tom", 8)  # 创建对象s1
s2 = Student_Grade("Jerry", 7)  # 创建对象s2

s1.print_grade()
s2.print_grade()
