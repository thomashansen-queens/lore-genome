"""
Reader plugin for Binary Alignment/Map (BAM) files.
"""
import lore
import pysam


@lore.reader(extensions=["bam"])
class BamReader(lore.BaseReader):
    """
    Reader for Binary Alignment/Map (BAM) files. BAM is a binary format for
    storing sequence data, often used in genomics.
    """
    def get_metadata(self) -> dict:
        base_meta = self.get_base_metadata()
        base_meta["can_stream"] = True
        return base_meta

    def stream(self, config: dict | None = None, **kwargs):
        """Memory-safe generator. Yields coverage pileup or raw reads"""
        io_config = {**(config or {}), **kwargs}
        mode = io_config.get("mode", "reads")  # 'coverage' or 'reads' (default)

        bam = pysam.AlignmentFile(self.path, "rb")

        # Get a target contig/chromosome to pile up on; default to first reference sequence
        contig = io_config.get("contig")
        start = io_config.get("start")
        stop = io_config.get("stop")

        if mode == "coverage":
            if not bam.has_index():
                raise RuntimeError(
                    f"Cannot compute coverage: BAM index (.bai) is missing "
                    f" or not adjacent to BAM file: '{self.path}'"
                )

            target_contig = contig or bam.references[0]

            for pileupcolumn in bam.pileup(target_contig, start=start, stop=stop, truncate=True):
                yield {
                    "position": pileupcolumn.reference_pos,
                    "depth": pileupcolumn.get_num_aligned(),
                }

        elif mode == "reads":
            if contig:
                if not bam.has_index():
                    raise RuntimeError(
                        f"Cannot fetch reads for contig '{contig}': BAM index (.bai) is missing "
                        f" or not adjacent to BAM file: '{self.path}'"
                    )
                read_iterator = bam.fetch(contig, start=start, stop=stop)
            else:
                read_iterator = bam.fetch(until_eof=True)

            for read in read_iterator:
                yield {
                    "id": read.query_name,
                    "contig": read.reference_name,
                    # Geometry
                    "start": read.reference_start,
                    "end": read.reference_end,
                    "is_reverse": read.is_reverse,
                    "query_length": read.query_length,
                    # CIGAR strings
                    "cigar": read.cigarstring,
                    "mapping_quality": read.mapping_quality,
                    # Mapping quality and flags
                    "is_unmapped": read.is_unmapped,
                }

        else:
            raise ValueError(f"Unknown BAM streaming mode: {mode}")
