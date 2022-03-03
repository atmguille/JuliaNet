compilar:
	julia -e 'using Pkg; Pkg.add("ArgParse")'

compila_mp:
	make compilar

ayuda_mp:
	julia FrioCalor.jl -h

ejecuta_mp:
	julia FrioCalor.jl --input_file input_mp.txt --output_file output_mp.txt 

ayuda_perceptron:
	julia Perceptron.jl -h

ejecuta_perceptron:
	julia Perceptron.jl --input_file problema_real1.txt --output_file prediccion_perceptron_prob1.txt --umbral 0.2 --tasa_aprendizaje 0.025 --max_epocas 50 --modo 1 --porcentaje 0.8

ayuda_adaline:
	julia Adaline.jl -h

ejecuta_adaline:
	julia Adaline.jl --input_file problema_real1.txt --output_file prediccion_adaline_prob1.txt --tasa_aprendizaje 0.04 --max_epocas 50 --tolerancia 0.02 --modo 1 --porcentaje 0.8