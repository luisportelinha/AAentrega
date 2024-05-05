using Flux
using WAV
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using JLD2, FileIO
using Statistics: mean
using DelimitedFiles
using Float8s
using Random


muestras_input=65536
clases=10
#train_imgs   = load("MNIST.jld2", "train_imgs");
#println(size(train_imgs))
labels = 1:clases; # Las etiquetas

function gen_input(ruta_archivo::AbstractString)
    # Leer el archivo .wav
    audio, sample_rate = WAV.wavread(ruta_archivo)
    
    # Normalizar la señal de audio
    audio = audio / maximum(abs.(audio))

    # Crear el input para la red neuronal (usando un arreglo unidimensional)
    input = reshape(audio, 1, :)
    return input
end

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
    tamano=size(array,2)
    # Copiar los valores del array original al nuevo array, desplazándolos
    for i in 1:veces
        if(i<=tamano)
            nuevo_array[i,1]=array[1,i]
        end
    end
    for i in 1:n_filas
            for j in 2:n_columnas-1
                j_aux=(j-1)*(veces*3/4)
                j_aux=round(Int,j_aux)
                nuevo_array[i,j] = array[1,i+j_aux]
            end
    end
    if(veces>tamano)
        veces=tamano
    end
    for i in 1:veces
        nuevo_array[i,n_columnas]=array[1,(length(array))-veces+i]
    end
    return nuevo_array
end

function FFT_data(input_wav1::String, value::Int, folder_names)
    # El target de debe cambiar
    #canales, frecuencia_muestreo, tasa_bits_codificacion, tamano = obtener_info_wav(input_wav1)
    input = gen_input(input_wav1)
    input1 = convertir_array(input, muestras_input)
        
    # Preinicializar las matrices con el tamaño adecuado
    input_size = size(input1, 2)
    targets = []

    for i in 1:input_size
        targets=push!(targets ,value)
    end
    return input1, targets
end

function procesar_archivos(input_folder::String)
    # Obtener la lista de carpetas en el directorio de entrada
    subfolders = readdir(input_folder)
    primero=true
    input = nothing
    target = nothing
    
    # Inicializar input y target
    
    value = 1
    #@assert(size(trainingInputs, 1) == size(trainingTargets, 1))
    #@assert(size(validationInputs, 1) == size(validationTargets, 1))
    #@assert(size(testInputs, 1) == size(testTargets, 1))
    folder_names = Float16[]
    for subfolder in subfolders
        push!(folder_names, parse(Float16, subfolder))
    end

    for subfolder in subfolders
        println(subfolder)
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

                        input, target = FFT_data(input_wav, value,folder_names)
                        #input = [round(x, digits=6) for x in input]


                    else
                        input_aux, target_aux = FFT_data(input_wav, value,folder_names)
                        input_aux = [round(x, digits=7) for x in input_aux]
                        input = hcat(input, input_aux)
                        target = vcat(target, target_aux)
                        
                    end

                    #println(target)                    

                else
                    println("No se encontró un archivo WAV correspondiente para $input_file")
                end
            end

        end
        value += 1
        println(size(input))
        open("input.txt", "w") do archivo
            writedlm(archivo, input)
        end
        open("target.txt", "w") do archivo
            writedlm(archivo, target)
        end
        #println(size(load_object("input.jld2");))
    end
    #return input,target
end



procesar_archivos("dataset5");

archivo = open("input.txt", "r")
train_imgs = readdlm(archivo)
close(archivo)

archivo = open("target.txt", "r")
target = readdlm(archivo)
close(archivo)


funcionTransferenciaCapasConvolucionales = relu;

function convertirArrayImagenesHWCN(imagenes)
    numPatrones = size(imagenes,2);
    nuevoArray = Array{Float32,3}(undef, muestras_input, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (length(imagenes[:,i])==(muestras_input)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,1,i] .= imagenes[:,i];
    end;
    return nuevoArray;
end;
N=size(train_imgs,2)
train_imgs = convertirArrayImagenesHWCN(train_imgs);


#N = 100;

println(N)



# Construimos la matriz tridimensional con las señales, donde cada dimensión es:
#  1 - La señal
#  2 - Canal (en este ejemplo sólo hay un canal, pero podría haber varias señales)
#  3 - El número de instancia
columnas_totales = size(train_imgs, 3)
indices = collect(1:columnas_totales)
Random.shuffle!(indices)

train=(columnas_totales-round(Int,columnas_totales/5))
test=round(Int,columnas_totales/5)
inputs = Array{Float32,3}(undef, muestras_input, 1, train);

inputs_test = Array{Float32,3}(undef, muestras_input, 1, test);
train_labels=target[indices[round(Int,columnas_totales/5+1):N]]
test_labels=target[indices[1:round(Int,columnas_totales/5)]]


for i in round(Int,columnas_totales/5+1):N
    inputs[:,1,i-round(Int,N/5)] .= reshape(train_imgs[:,1,indices[i]], muestras_input);

end;
for i in 1:round(Int,columnas_totales/5)
    inputs_test[:,1,i] .= reshape(train_imgs[:,1,indices[i]], muestras_input);
end;

println("Tamaño de la matriz de entrenamiento: ", size(inputs))
println("   Longitud de la señal: ", size(inputs,1))
println("   Numero de instancias: ", size(inputs,3))

ann = Chain(
    Conv((3,), 1=>1, pad=1, funcionTransferenciaCapasConvolucionales),
    MaxPool((2,)),
    x -> reshape(x, :, size(x, 3)),
    Dense(32768, 10),
    softmax
);

println("Tamaño de las salidas: ", size(ann(inputs)));




# Vamos a probar la RNA capa por capa y poner algunos datos de cada capa
# Usaremos como entrada varios patrones de un batch
numBatchCoger = 1; numImagenEnEseBatch = [1, 6];
println(size(inputs))
# Para coger esos patrones de ese batch:
#  train_set es un array de tuplas (una tupla por batch), donde, en cada tupla, el primer elemento son las entradas y el segundo las salidas deseadas
#  Por tanto:
#   train_set[numBatchCoger] -> La tupla del batch seleccionado
#   train_set[numBatchCoger][1] -> El primer elemento de esa tupla, es decir, las entradas de ese batch
#   train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch] -> Los patrones seleccionados de las entradas de ese batch
entradaCapa = inputs[:,:,numImagenEnEseBatch];
numCapas = length(Flux.params(ann));

println("La RNA tiene ", numCapas, " capas:");
for numCapa in 1:numCapas
    println("   Capa ", numCapa, ": ", ann[numCapa]);
    # Le pasamos la entrada a esta capa
    global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
    capa = ann[numCapa];
    println(size(entradaCapa))
    salidaCapa = capa(entradaCapa);
    println("      La salida de esta capa tiene dimension ", size(salidaCapa));
    entradaCapa = salidaCapa;

end
# Sin embargo, para aplicar un patron no hace falta hacer todo eso.
#  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
ann(inputs[:,:,numImagenEnEseBatch]);

batch_size = 80
# Creamos los indices: partimos el vector 1:N en grupos de batch_size
gruposIndicesBatch = Iterators.partition(1:size(inputs,3), batch_size);
println("Se han creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");


# Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
#  Como primer elemento, las imagenes de ese batch
#     train_imgs[:,:,:,indicesBatch]
#  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
#     Para conseguir estas salidas deseadas, se hace una llamada a la funcion onehotbatch, que realiza un one-hot-encoding de las etiquetas que se le pasen como parametros
#     onehotbatch(train_labels[indicesBatch], labels)
#  Por tanto, cada batch será un par dado por
#     (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels))
# Sólo resta iterar por cada batch para construir el vector de batches
train_set = [ (inputs[:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];

# Creamos un batch similar, pero con todas las imagenes de test
test_set = (inputs_test, onehotbatch(test_labels, labels));


# Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
train_imgs = nothing;
test_imgs = nothing;
GC.gc(); # Pasar el recolector de basura




# Sin embargo, para aplicar un patron no hace falta hacer todo eso.
#  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
ann(train_set[numBatchCoger][1][:,:,numImagenEnEseBatch]);

# Definimos la funcion de loss de forma similar a las prácticas de la asignatura
loss(ann, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

# Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
# Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


# Mostramos la precision antes de comenzar el entrenamiento:
#  train_set es un array de batches
#  accuracy recibe como parametro un batch
#  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
#   y devuelve un array con los resultados
#  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
#   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy.(train_set)), " %");



# Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
eta = 0.01;
opt_state = Flux.setup(Adam(eta), ann);


println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;

while !criterioFin

    # Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
    global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

    # Se entrena un ciclo
    Flux.train!(loss, ann, train_set, opt_state);

    numCiclo += 1;

    # Se calcula la precision en el conjunto de entrenamiento:
    precisionEntrenamiento = mean(accuracy.(train_set));
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

    # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
    if (precisionEntrenamiento >= mejorPrecision)
        mejorPrecision = precisionEntrenamiento;
        precisionTest = accuracy(test_set);
        println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
        mejorModelo = deepcopy(ann);
        numCicloUltimaMejora = numCiclo;
    end

    #da error
    # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
    if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-6)
        global eta
        eta /= 10.0
        println("   No se ha mejorado la precision en el conjunto de entrenamiento en 5 ciclos, se baja la tasa de aprendizaje a ", eta);
        adjust!(opt_state, eta)
        numCicloUltimaMejora = numCiclo;
    end

    # Criterios de parada:

    # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
    if (precisionEntrenamiento >= 0.9)
        println("   Se para el entenamiento por haber llegado a una precision de 90%")
        criterioFin = true;
    end

    # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
    if (numCiclo - numCicloUltimaMejora >= 10)
        println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
        criterioFin = true;
    end
end
