"""
Adapter to tabulate 
"""
import lore


@lore.adapter()
class BamAdapter(lore.TabularAdapter):
    """
    Adapter for Binary Alignment/Map (BAM) files. BAM is a binary format for
    storing sequence data, often used in genomics.
    """
    accepted_formats = {"bam"}
    accepted_types = {"*"}

    @property
    def schema(self):
        return {
            "id": "id",
            "contig": "contig",
            "start": "start",
            "end": "end",
            "length": "query_length",
            "is_reverse": "is_reverse",
            "mate_is_reverse": "mate_is_reverse",
            "mapping_quality": "mapping_quality",
            "is_paired": "is_paired",
            "is_proper_pair": "is_proper_pair",
            "is_unmapped": "is_unmapped",
        }
