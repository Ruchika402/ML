import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
students = pd.read_csv(r"C:\Users\adakr\OneDrive\Desktop\students_data.csv")


students.plot(x="Student")

plt.title("All Data Plot")
plt.show()   


# Create bar chart
plt.bar(students["Student"], students["Marks"])

# Labels and title
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Students Marks Chart")

# Show graph
plt.show()

#pie chart
plt.pie(students["Marks"],labels=students["Student"],)

plt.title("Marks Distribution")
plt.show()


#line chart
plt.plot(students["Student"],students["Marks"],)

plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Students Marks Line Chart")

plt.show()


#scatter plot
plt.scatter(students["Hours_Studied"],students["Marks"])

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours Studied vs Marks")

plt.show()


#histogram
plt.hist(students["Marks"])

plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Marks Histogram")

plt.show()



