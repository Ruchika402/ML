import pandas as pd
import numpy as np
'''
s = pd.Series([10,20,30,40,50], index = ['a','b','c','d','e'], dtype = int)
print(s)
print(s.values)
print(s.index)
dict = {"a":100,"b":200,"c":300}
x=pd.Series(dict)
'''
data_array = np.array(
    [
        [1,"R",25],
        [2,"A",30],
        [3,"S",22]
    ]
)
x = pd.DataFrame(data_array,columns=["ID","Name","Age"])
print(x)
'''
array1 = np.array(
    [
        [1,"Ruchika",25,88],
        [2,"Anujyoti",30,92],
        [3,"Khushi",22,79],
        [4,"Aritra",28,85],
        [5,"Priyangshu",35,95]
    ]
)
x1 = pd.DataFrame(array1,columns=["ID","Name","Age","Marks"])
print(x1)

'''
df = pd.read_csv(r"C:\Users\adakr\OneDrive\Desktop\students.csv")
print(df)
print(df.columns.tolist())
print(df.shape)
print(df.size)
print(df.index)
print(df.dtypes)
print(df.info())



import pandas as pd
x = pd.read_csv(r"C:\Users\adakr\OneDrive\Desktop\Students (1).csv")
print(x)

print(x[['Name', 'Marks']])#name and marks

high_scorers = x[x['Marks'] > 85]
print(high_scorers)#students score than 85

kolkata_students = x[x['City'] == 'Kolkata']
print(kolkata_students)#details of students from kolkata

average_marks = x['Marks'].mean()
print(f"Average Marks: {average_marks:.2f}")#average marks of all students

max_marks = x['Marks'].max()
highest_scorers = x[x['Marks'] == max_marks]
print(highest_scorers)#students who score highest marks

city_counts = x['City'].value_counts()
print(city_counts)#count number of student in each city

x['Status'] = x['Marks'].apply(lambda m: 'Pass' if m >= 80 else 'Fail')
print(x[['Name', 'Marks', 'Status']])#Create a new column called Status (Marks ≥ 80 → "Pass", Marks < 80 → "Fail")

sorted_students = x.sort_values('Marks', ascending=False)
print(sorted_students[['Name', 'Marks', 'Status']].head(3))#Sort the DataFrame by Marks in descending order and display top 3 students

gender_avg = x.groupby('Gender')['Marks'].mean()
print(gender_avg)#Find the average marks of students grouped by Gender

print(x)# Display the final DataFrame with all changes



