using WAV
using Flux
using SampledSignals
using CSV
using DataFrames
using FFTW
using Statistics
using Plots
using Flux.Losses
using ScikitLearn
using LinearAlgebra
using Random
@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier


muestras_input=65536#potencia de 2 mayor o igual que cuatro
output_length=2
input_length=2


#Saca en un array los datos de .wav
function gen_input(ruta_archivo::AbstractString)
    # Leer el archivo .wav
    audio, sample_rate = WAV.wavread(ruta_archivo)
    
    # Normalizar la señal de audio
    audio = audio / maximum(abs.(audio))

    # Crear el input para la red neuronal (usando un arreglo unidimensional)
    input = reshape(audio, 1, :)
    return input
end

function obtener_info_wav(archivo)
    # Leer el archivo WAV
    audio, samplerate = wavread(archivo)
    
    # Obtener información del archivo WAV
    canales = size(audio, 2)
    frecuencia_muestreo = samplerate
    tasa_bits_muestra = eltype(audio) == Float32 ? 32 : 16  # Tasa de bits por muestra (bits)
    tamano = size(audio, 1) 
    # Calcular la tasa de bits de codificación (kbps)
    tasa_bits_codificacion = tasa_bits_muestra * frecuencia_muestreo * canales / 1000
    
    return canales, frecuencia_muestreo, tasa_bits_codificacion, tamano
end

#Convierte el array de datos del .wav en una matriz con un historico de muestras_input valores
function convertir_array(array::Matrix{T}, veces::Int) where T
    # Calcular el tamaño del nuevo array
    n_filas = veces
    n_columnas = (length(array))/(veces)
    n_columnas=round(Int,n_columnas)
    if (length(array))-veces/4>(n_columnas-1)*(veces*3/4)+(veces*3/4)
        n_columnas=n_columnas+1
    end
    
    # Crear un nuevo array bidimensional con ceros
    nuevo_array = zeros(T, n_filas, n_columnas)
    
    # Copiar los valores del array original al nuevo array, desplazándolos
    for i in 1:veces
        nuevo_array[i,1]=array[1,i]
    end
    for i in 1:n_filas
            for j in 2:n_columnas-1
                j_aux=(j-1)*(veces*3/4)
                j_aux=round(Int,j_aux)
                nuevo_array[i,j] = array[1,i+j_aux]
            end
    end
    for i in 1:veces
        nuevo_array[i,n_columnas]=array[1,(length(array))-veces+i]
    end
    return nuevo_array
end

function crear_wav(data::Vector{Float64}, fs::Int, filename::String)
    wavwrite(data, fs, filename)
end

#Crea una matriz que para cada muestra guarda mediate un 0 o un 1 que nota esta sonando que cada valor de cada muestra
function gen_target(filename::String, num_filas::Int, veces::Int)
    # Leer el archivo CSV
    df = CSV.File(filename) |> DataFrame
    valores=127#-recortar_valores
    matriz = zeros(Int, valores, num_filas)
    num_filas=num_filas#-(muestras_input)/2
    num_filas=round(Int,num_filas)
    #salto=round(Int,muestras_input*2/3)
    # Iterar sobre cada fila del DataFrame
    for fila in 1:size(df, 1)
        # Obtener los valores de start_time, end_time y note
        inicio = df[fila, :start_time]
        fin = df[fila, :end_time]
        columna = df[fila, :note]
        columna=columna#-recortar_valores
        if columna>0
            # Iterar sobre las filas y poner 1 en las columnas correspondientes
            for num_fila in inicio:fin
                if 1 <= num_fila <= num_filas
                    fila_real=(num_fila-1-(num_fila-1)%veces)/veces+1
                    fila_real=round(Int,fila_real)
                    matriz[ columna,fila_real] = 1
                    fila_real=(num_fila-(num_fila+veces*3/4)%veces+veces*3/4)/veces+1
                    fila_real=round(Int,fila_real)
                    #if fila_real>0
                    matriz[ columna,fila_real] = 1
                    #end
                end
            end 
        end
    end
    return matriz
end

function convertir_a_array(array::Vector{T}, veces::Int) where T
    # Calcular el tamaño del nuevo array
    n_filas = 1
    n_columnas = ceil(Int, length(array) / veces)
    
    # Crear un nuevo array bidimensional con ceros
    nuevo_array = zeros(T, n_filas, n_columnas)
    
    # Copiar los valores del array original al nuevo array
    for i in 1:n_columnas
        inicio = (i - 1) * veces + 1
        fin = min(i * veces, length(array))
        nuevo_array[1, i] = mean(@view(array[inicio:fin]))
    end
    
    return nuevo_array
end

function procesar_archivos(input_folder::String)
    # Obtener la lista de carpetas en el directorio de entrada
    subfolders = readdir(input_folder)
    
    # Inicializar input y target
    input = nothing
    target = nothing
    value = 1
    #@assert(size(trainingInputs, 1) == size(trainingTargets, 1))
    #@assert(size(validationInputs, 1) == size(validationTargets, 1))
    #@assert(size(testInputs, 1) == size(testTargets, 1))

    for subfolder in subfolders
        
        subfolder_path = joinpath(input_folder, subfolder)
        
        # Verificar si el elemento es una carpeta
        if isdir(subfolder_path)
            # Obtener la lista de archivos en la subcarpeta actual
            input_files = readdir(subfolder_path)
            input_files = filter(x -> endswith(x, ".wav"), input_files)
            
            for input_file in input_files
                # Verificar si hay un archivo WAV correspondiente
                filename_base = splitext(input_file)[1]  # Obtener el nombre base sin la extensión
                matching_wav = joinpath(subfolder_path, filename_base * ".wav")
                
                if isfile(matching_wav)
                    # Si hay un archivo WAV correspondiente, cargar ambos archivos
                    input_wav = matching_wav
                    target_wav = matching_wav  # No hay target en el código proporcionado

                    # Aquí puedes realizar tu tarea de procesamiento de datos
                    if input === nothing && target === nothing
                        input_aux, target_aux = FFT_data(input_wav, value)
                        input = input_aux
                        target = target_aux
                    else
                        input_aux, target_aux = FFT_data(input_wav, value)
                        input = hcat(input, input_aux)
                        target = hcat(target, target_aux)
                    end

                    #println(target)                    

                else
                    println("No se encontró un archivo WAV correspondiente para $input_file")
                end
            end

        end
        value += 1
    end
    println()
    println("-RRNNAA:")
    lose=entrenar_RRNNAA(input, target)
    println("Error RRNNAA: ",lose)
    println()
    println("-SMV:")
    mse,mae=entrenar_svm(input, target)
    println("SVM MSE:",mse)
    println("SVM MAE:",mae)
    println()
    println("-TREE:")
    mse,mae=entrenar_tree(input, target)
    println("TREE MSE:",mse)
    println("TREE MAE:",mae)
    println()
    println("-KNe:")
    mse,mae=entrenar_KNe(input, target)
    println("KNe MSE:",mse)
    println("KNe MAE:",mae)
end

function entrenar_RRNNAA(input_train,target_train)
    gr();
    minLoss=0.1
    learningRate=0.02
    input_validation=nothing
    target_validation=nothing
    input_test=nothing
    target_test=nothing
    max_ciclos=120
    iteracion=1
    restar=0
    sin_mejora=0
    parar_nomejora=20
    historico_train=Float64[]
    historico_validation=Float64[]
    historico_test=Float64[]
    #println(size(input_train,2))
    #extrae del conjunto de entrenamiento el conjunto de test y validacion
    for i in 1:size(input_train,2)
        x=rand(1:10)
        if x%10==2||x%10==6
            if input_validation==nothing
                input_validation=zeros(input_length,1)
                input_train,input_validation[:,1]=extract_and_remove(input_train,i-restar)
            else
                aux=zeros(input_length,1)
                input_train,aux[:,1]=extract_and_remove(input_train,i-restar)
                input_validation=hcat(input_validation,aux)
            end
            if target_validation==nothing
                target_validation=zeros(output_length,1)
                target_train,target_validation[:,1]=extract_and_remove(target_train,i-restar)
            else
                aux=zeros(output_length,1)
                target_train,aux[:,1]=extract_and_remove(target_train,i-restar)
                target_validation=hcat(target_validation,aux)
            end
            restar+=1
        end
        if x%10==3
            if input_test==nothing
                input_test=zeros(input_length,1)
                input_train,input_test[:,1]=extract_and_remove(input_train,i-restar)
            else
                aux=zeros(input_length,1)
                input_train,aux[:,1]=extract_and_remove(input_train,i-restar)
                input_test=hcat(input_test,aux)
            end
            if target_test==nothing
                target_test=zeros(output_length,1)
                target_train,target_test[:,1]=extract_and_remove(target_train,i-restar)
            else
                aux=zeros(output_length,1)
                target_train,aux[:,1]=extract_and_remove(target_train,i-restar)
                target_test=hcat(target_test,aux)
            end
            restar+=1
        end
    end
    ann = Chain(
        Dense(input_length, 4, σ),
        #Dense(5, 4, σ),
        Dense(4, 3, σ),
        Dense(3, output_length, identity),softmax );
    
    loss(model,x, y) = Losses.crossentropy(model(x), y)    
    opt_state = Flux.setup(Adam(learningRate), ann)    
    outputP = ann(input_train)
    vlose = loss(ann,input_validation, target_validation)
    mejor=vlose
    #while (vlose > minLoss&&max_ciclos>iteracion&&sin_mejora!=parar_nomejora)
    while (vlose > minLoss&&max_ciclos>iteracion)

        Flux.train!(loss, ann, [(input_train, target_train)], opt_state)  
        vlose = loss(ann,input_validation, target_validation)
        outputP = ann(input_validation)
        #vacc = accuracy(outputP, target_validation)
        #if(vlose>mejor)
        #    sin_mejora+=1
        #else
        #    mejor=vlose
        #    sin_mejora=0
        #end
        push!(historico_train,loss(ann,input_train, target_train))
        push!(historico_test,loss(ann,input_test, target_test))
        push!(historico_validation,vlose)
        #println(vlose)
        iteracion+=1
    end
    
    vlose = loss(ann,input_test, target_test)
    outputP = ann(input_test)
    #vacc = accuracy(outputP, target_test)
    push!(historico_train,loss(ann,input_train, target_train))
    push!(historico_test,loss(ann,input_test, target_test))
    push!(historico_validation,vlose)
    p1=plot(historico_train, title="Historico Train", subplot=1)
    p2=plot(historico_test, title="Historico Test", subplot=1)
    p3=plot(historico_validation, title="Historico Validation", subplot=1)
    display(plot(p1,p2,p3, layout = (3,1)));   
    # Convertir los valores en bool dependiendo si son mayores o menores que 0.5
    outputP = nearest_to_one_matrix(outputP)
    outputP = Array{Bool}(outputP .> 0.5)
    # Convertir los valores en bool dependiendo si son mayores o menores que 0.5
    target_test = nearest_to_one_matrix(target_test)
    target_test = Array{Bool}(target_test .> 0.5)
    printConfusionMatrix(outputP'[:,1],target_test'[:,1])
    

    return vlose
end

function nearest_to_one_matrix(matrix::Matrix)
    result = similar(matrix, Bool)

    for j in 1:size(matrix, 2)
        min_distance = Inf
        nearest_index = -1

        # Encontrar el índice del valor más cercano a 1 en la columna j
        for i in 1:size(matrix, 1)
            distance = abs(1 - matrix[i, j])
            if distance < min_distance
                min_distance = distance
                nearest_index = i
            end
        end

        # Configurar a 1 el valor más cercano a 1 y el resto a 0 en la columna j
        result[nearest_index, j] = 1
        for i in 1:size(matrix, 1)
            if i != nearest_index
                result[i, j] = 0
            end
        end
    end

    return result
end

function entrenar_svm(input_train, target_train)
    input_test=nothing
    target_test=nothing
    restar=0
    for i in 1:size(input_train,2)
        x=rand(1:10)
        if x%10==2
            if input_test==nothing
                input_test=zeros(input_length,1)
                input_train,input_test[:,1]=extract_and_remove(input_train,i-restar)
            else
                aux=zeros(input_length,1)
                input_train,aux[:,1]=extract_and_remove(input_train,i-restar)
                input_test=hcat(input_test,aux)
            end
            if target_test==nothing
                target_test=zeros(output_length,1)
                target_train,target_test[:,1]=extract_and_remove(target_train,i-restar)
            else
                aux=zeros(output_length,1)
                target_train,aux[:,1]=extract_and_remove(target_train,i-restar)
                target_test=hcat(target_test,aux)
            end
            restar+=1
        end
        
    end
    model = SVC(kernel="rbf", degree=3, gamma=1, C=3);
    fit!(model, input_train', crear_vector(target_train)');
    testOutputs = predict(model, input_test');
    #aux=crear_vector(target_test)'
    #println(aux)
    #printConfusionMatrix(testOutputs,collect(aux))
    
    aux=(Array{Bool}(target_test .== 1))'
    printConfusionMatrix(recrear_vector(testOutputs,output_length)[:,1],aux[:,1])
    # Calcular el error cuadrático medio (MSE)
    mse = mean((testOutputs .- target_test').^2)
    # Calcular el error absoluto medio (MAE)
    mae = mean(abs.(testOutputs .- target_test'))
    return mse,mae    
end

function entrenar_tree(input_train, target_train)
    input_test=nothing
    target_test=nothing
    restar=0
    for i in 1:size(input_train,2)
        x=rand(1:10)
        if x%10==2
            if input_test==nothing
                input_test=zeros(input_length,1)
                input_train,input_test[:,1]=extract_and_remove(input_train,i-restar)
            else
                aux=zeros(input_length,1)
                input_train,aux[:,1]=extract_and_remove(input_train,i-restar)
                input_test=hcat(input_test,aux)
            end
            if target_test==nothing
                target_test=zeros(output_length,1)
                target_train,target_test[:,1]=extract_and_remove(target_train,i-restar)
            else
                aux=zeros(output_length,1)
                target_train,aux[:,1]=extract_and_remove(target_train,i-restar)
                target_test=hcat(target_test,aux)
            end
            restar+=1
        end
    end
    model = DecisionTreeClassifier(max_depth=2, random_state=1)
    fit!(model, input_train', crear_vector(target_train)');
    testOutputs = predict(model, input_test');
    aux=(Array{Bool}(target_test .> 0.5))'
    printConfusionMatrix(recrear_vector(testOutputs,output_length)[:,1],aux[:,1])
    # Calcular el error cuadrático medio (MSE)
    mse = mean((testOutputs .- target_test').^2)
    # Calcular el error absoluto medio (MAE)
    mae = mean(abs.(testOutputs .- target_test'))
    return mse,mae
end

function entrenar_KNe(input_train, target_train)
    input_test=nothing
    target_test=nothing
    restar=0
    for i in 1:size(input_train,2)
        x=rand(1:10)
        if x%10==2
            if input_test==nothing
                input_test=zeros(input_length,1)
                input_train,input_test[:,1]=extract_and_remove(input_train,i-restar)
            else
                aux=zeros(input_length,1)
                input_train,aux[:,1]=extract_and_remove(input_train,i-restar)
                input_test=hcat(input_test,aux)
            end
            if target_test==nothing
                target_test=zeros(output_length,1)
                target_train,target_test[:,1]=extract_and_remove(target_train,i-restar)
            else
                aux=zeros(output_length,1)
                target_train,aux[:,1]=extract_and_remove(target_train,i-restar)
                target_test=hcat(target_test,aux)
            end
            restar+=1
        end
        
    end
    model = KNeighborsClassifier(2);
    fit!(model, input_train', crear_vector(target_train)');
    testOutputs = predict(model, input_test');
    aux=(Array{Bool}(target_test .> 0.5))'
    printConfusionMatrix(recrear_vector(testOutputs,output_length)[:,1],aux[:,1])
    # Calcular el error cuadrático medio (MSE)
    mse = mean((testOutputs .- target_test').^2)
    # Calcular el error absoluto medio (MAE)
    mae = mean(abs.(testOutputs .- target_test'))
    return mse,mae
end

function crear_vector(matriz::Matrix{Float64})
    m, n = size(matriz)
    vector_resultado = zeros(Int, n)
    
    for i in 1:m
        for j in 1:n
            if matriz[i, j] == 1
                vector_resultado[j] = i
            end
        end
    end
    
    return vector_resultado
end

function recrear_vector(array,size::Int)
    #println(array)
    result=zeros(length(array),size)
    for i in 1:length(array)
        result[i,array[i]]=1
    end
    #println(result)
    return Array{Bool}(result .> 0.5)
    #return Array{Bool}(array .> 0.5)
end

function FFT_data(input_wav1::String, value::Int)
    # El target de debe cambiar
    canales, frecuencia_muestreo, tasa_bits_codificacion, tamano = obtener_info_wav(input_wav1)
    input = gen_input(input_wav1)
    input1 = convertir_array(input, muestras_input)
        
    # Preinicializar las matrices con el tamaño adecuado
    input_size = size(input1, input_length)
    inputs = Matrix{Float64}(undef, input_length, input_size)
    targets = Matrix{Float64}(undef, output_length, input_size)

    for i in 1:input_size
        aux = reshape(input1[:, i], 1, length(input1[:, i]))
        input_aux = zeros(input_length, 1)
        input_aux[1, 1], input_aux[input_length, 1] = mirar2notas(aux, frecuencia_muestreo, 0, 0)
        target_aux = zeros(output_length, 1)
        target_aux[value, 1] = 1.0
        inputs[:, i] .= input_aux
        targets[:, i] .= target_aux
    end

    return inputs, targets
end

function extract_and_remove(matrix::AbstractMatrix, col_index::Integer)
    if 1 <= col_index <= size(matrix, 2)
        col = matrix[:, col_index]
        matrix = hcat(matrix[:, 1:col_index-1], matrix[:, col_index+1:end])
        #println(size(matrix))
        return matrix, col
    else
        throw(ArgumentError("Índice de columna fuera de rango"))
    end
end

function mirar2notas(input::Matrix{Float64},frecuencia::Float32,note1::Int, note2::Int)

    #canales, Fs, tasa_bits_codificacion,tamano = obtener_info_wav(input_wav)
    #input_org = gen_input(input_wav)
    #input=convertir_array(input_org, muestras_input)
    #println(size(input_org,2))
    # Numero de muestras
    
    n = size(input,2);
    # Que frecuenicas queremos coger
    f1 = note1; f2 = note2;
    Fs=frecuencia;

    #println("$(n) muestras con una frecuencia de $(Fs) muestras/seg: $(n/Fs) seg.")

    # Creamos una señal de n muestras: es un array de flotantes
    x = 1:n;
    senalTiempo = input[1,:];
    
    
    # Representamos la señal
    #plotlyjs();
    #graficaTiempo = plot(x, senalTiempo, label = "", xaxis = x);
    
    # Hallamos la FFT y tomamos el valor absoluto
    senalFrecuencia = abs.(fft(senalTiempo));
    
    
    
    # Los valores absolutos de la primera mitad de la señal deberian de ser iguales a los de la segunda mitad, salvo errores de redondeo
    # Esto se puede ver en la grafica:
    #graficaFrecuencia = plot(senalFrecuencia, label = "");
    #  pero ademas lo comprobamos en el codigo
    if (iseven(n))
        @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
    else
        @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
    end;
    
    # Grafica con la primera mitad de la frecuencia:
    #graficaFrecuenciaMitad = plot(senalFrecuencia, label = "");
    
    
    # Representamos las 3 graficas juntas
    #display(plot(graficaTiempo, graficaFrecuencia, graficaFrecuenciaMitad, layout = (3,1)));
    
    
    # A que muestras se corresponden las frecuencias indicadas
    #  Como limite se puede tomar la mitad de la frecuencia de muestreo

    # recortamos la mitad no necesaria
    f1 = 1; f2 =Int(round(length(senalFrecuencia)/2));

    m1 = Int(round(f1*2*length(senalFrecuencia)/Fs));
    m2 = Int(round(f2*2*length(senalFrecuencia)/Fs));
    
    # Unas caracteristicas en esa banda de frecuencias
    #println("Media de la señal en frecuencia entre $(f1) y $(f2) Hz: ", mean(senalFrecuencia[m1:m2]));
    #println("Desv tipica de la señal en frecuencia entre $(f1) y $(f2) Hz: ", std(senalFrecuencia[m1:m2]));
    return mean(senalFrecuencia[m1:m2]),std(senalFrecuencia[m1:m2])
end

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    @assert size(targets, 2) == size(outputs, 2) "The number of columns in targets and outputs must be the same."

    # Handle single-column case directly
    if size(targets, 2) == 1
        return mean(targets[:] .== outputs[:])
    end

    # Calculate accuracy for multi-class outputs
    correctClassifications = all(targets .== outputs, dims=2)
    return mean(correctClassifications)
end

function calculateMetrics(TN, FP, FN, TP)
    total = TN + FP + FN + TP
    acc = total == 0 ? 0.0 : (TN + TP) / total
    errorRate = 1.0 - acc

    metrics = Dict(
        "Recall" => FN + TP == 0 ? 0.0 : TP / (TP + FN),
        "Specificity" => TN + FP == 0 ? 0.0 : TN / (TN + FP),
        "Precision" => TP + FP == 0 ? 0.0 : TP / (TP + FP),
        "NPV" => TN + FN == 0 ? 0.0 : TN / (TN + FN),
        "F1" => TP + FP == 0 || FN + TP == 0 ? 0.0 : 2 * TP / (2 * TP + FP + FN)
    )

    return acc, errorRate, metrics
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TN = sum(.!outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    TP = sum(outputs .& targets)
    FP = sum(outputs .& .!targets)

    confMatrix = [TN FP; FN TP]
    acc, errorRate, metrics = calculateMetrics(TN, FP, FN, TP)

    return (acc, errorRate, metrics, confMatrix)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)

    println("Confusion Matrix:")
    println(confMatrix)
    println("Accuracy: $acc")
    println("Error Rate: $errorRate")
    println("Recall (Sensitivity): $recall")
    println("Specificity: $specificity")
    println("Precision (Positive Predictive Value): $precision")
    println("Negative Predictive Value: $NPV")
    println("F1 Score: $F1")
    println(" ")
end

Random.seed!(1234)

procesar_archivos("dataset1");