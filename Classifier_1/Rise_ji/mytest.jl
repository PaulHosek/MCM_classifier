

using GraphicalModelLearning

function create_histogram(file_path)
    # Initialize a dictionary to count occurrences of binary strings
    counts = Dict{String, Int}()
    
    # Open the file and read line by line
    open(file_path, "r") do file
        for line in eachline(file)
            # Remove newline characters and whitespace
            binary_string = strip(line)
            # Increment the count for the binary string
            counts[binary_string] = get(counts, binary_string, 0) + 1
        end
    end
    
    # Determine the maximum length of binary strings for column alignment
    max_length = maximum(length.(keys(counts)))
    
    # Convert the dictionary to an array with individual bits as columns
    histogram = []
    for (binary_string, count) in counts
        # Split the binary string into its bits and pad with zeros if necessary
        bits = lpad(binary_string, max_length, '0') |> collect |> x -> parse.(Int, x) .|> y -> y == 0 ? -1 : 1
        
        # Prepend the count to the bits array
        row = [count; bits]
        push!(histogram, row)
    end
    
    # Sort the histogram by count, descending
    sort!(histogram, by=x->x[1], rev=true)
    
    return histogram
end


file_path = "./test/data/trivial.dat" # Replace with your actual file path
histogram = create_histogram(file_path)
println(histogram)


# if abspath(PROGRAM_FILE) == @__FILE__
model = FactorGraph([0.0 0.1 0.2; 0.1 0.0 0.3; 0.2 0.3 0.0])
samples = sample(model, 100000)
println(samples)



# learned = learn(samples)

# err = abs.(convert(Array{Float64,2}, model) - learned)
# println(err)

