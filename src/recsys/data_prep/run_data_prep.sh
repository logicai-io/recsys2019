echo "Joining datasets"
python join_datasets.py
echo "Converting item metadata"
python convert_item_metadata_to_sets.py
echo "Extracting hotel dense features"
python extract_hotel_dense_features.py
echo "Extracting item prices"
python extract_item_prices.py
echo "Generating click indices"
python generate_click_indices.py
echo "Assigning poi to items"
python assign_poi_to_items.py
echo "Extracting city prices"
python extract_city_prices_percentiles.py
