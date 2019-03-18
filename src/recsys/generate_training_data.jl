using CSV, DataFrames, DataStructures


println("Loading")
train = CSV.read("../../data/train_sample_5.csv")

println("Sorting")
sort!(train, (:timestamp, :user_id, :step))

println("Iterating")
item_impressions_counter = DefaultDict{String, Int}(0)
for i in 1:size(train, 1)
    row = train[i, :]
    if row.action_type == "clickout item"
        global item_impressions_counter
        impressions = split(row.impressions,"|")
        for item_id in impressions
            item_impressions_counter[item_id] += 1
        end
    end
end