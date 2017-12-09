i = 1
from datetime import datetime
start_time = datetime.now()
print('hi')
while (datetime.now()-start_time).seconds<3*60:
	i+=1


file = open('time_test.txt','w') 
file.write('Congratulations, the project is done! It took: ' + str(i)+ '     '+ str((datetime.now()-start_time).seconds))
 
file.close() 

