compilar:
	julia -e 'using Pkg; Pkg.add("ArgParse")'

ayuda_retro:
	julia Backpropagation.jl -h

ejecuta_problema_real2:
	julia Backpropagation.jl --input_file data\problema_real2.txt --output_name problema_real2_0.01_10_500 --tasa_aprendizaje 0.01 --epocas 500 --modo 1 --red_config [10] --porcentaje 0.7

ejecuta_problema_real6:
	julia Backpropagation.jl --input_file data\problema_real6.txt --output_name problema_real6_0.1_20_5000_norm --tasa_aprendizaje 0.1 --epocas 5000 --modo 1 --red_config [20] --porcentaje 0.7 --normalizar
