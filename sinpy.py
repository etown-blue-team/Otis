import math

x = 0

while x <= 2*math.pi:
	print(str(x) + ": " + str(math.sin(x)))
	x += 0.05
	
	