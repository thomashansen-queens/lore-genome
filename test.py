import pickle
import pandas

# resolved_inputs, input_artifacts = pickle.load(open("test", "rb"))
# item_data = pickle.load(open("preview_result_adapter_explore", "rb"))
# item_data = pickle.load(open("item_data_materializer_adapted134", "rb"))
# item_data = pickle.load(open("resolved_inputs_materializer_adapted134", "rb"))
# item_data = pickle.load(open("results_preview_adapted_materializer", "rb"))
# ctx = pickle.load(open("ctx_preview_adapted_materializer", "rb"))
# payload = pickle.load(open("payload_preview_adapted_contextpy", "rb"))
# data = pickle.load(open("content_materialize_content_contextpy", "rb"))
thing = pickle.load(open("thing", "rb")) 

print(", ".join(thing))


# for x in payload[0]:
#     print(x)

# print(item_data)

# for x in item_data['source']:
#     print(x)
    # break


# for x in resolved_inputs['source']:
#     print(x)
#     break