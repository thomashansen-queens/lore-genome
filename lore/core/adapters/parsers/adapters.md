Adapters are reader classes used by LoRe.
All adapters are subclasses of the BaseAdapter ABC (abstract base class).
To take advantage of LoRe's magic materializer data slicing, use TabularAdapters with a defined schema.

Tabular adapters take some input, and output a list of records (list[dict]).
Unlress overridden, the `adapt()` method in a convenience function that chains together two separate processes:
1. Parse - Uses the `parse()` method to make a list of raw records
2. Adapts - Uses the `adapt_record()` method on each of those raw records to create an adapted record

This can be useful to tasks that need access to both. For example, filtering and sampling records can be done by:
1. Passing the raw data (load_as=lore.RAW)
2. Parsing it into records (parsed=adapter.parse(raw_data))
3. Adapting those records in the task handler (adapted = adapter.adapt(parsed))
4. Applying filtering or sampling to that adapted data, obtaining the indices of those records that passed the filter (final_records)
5. Serialize the parsed *but not adapted* records as output

For very large files, Adapters having streaming methods. Rather than calling `adapt()`, use `adapt_stream()`. To replicate the above style of access to both parsed and adapted records:
1. Set the task input to request a file strem (`load_as=lore.RAW_STREAM`)
2. Set up a generator to parse the raw stream (parsed = adapter.parse_stream(raw_stream))
3. Iterator over the generator applying the adapter's `adapt_record()` method on each element
4. Within the same loop, stream the result back to disk using the same `serialize()` method on the parsed *but not adapted* record
