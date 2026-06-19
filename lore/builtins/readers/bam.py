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
        mode = io_config.get("mode", "coverage")  # 'coverage' or 'reads'

        bam = pysam.AlignmentFile(self.path, "rb")

        # Get a target contig/chromosome to pile up on; default to first reference seuqence
        contig = io_config.get("contig", bam.references[0])
        start = io_config.get("start")
        stop = io_config.get("stop")

        if mode == "coverage":
            for pileupcolumn in bam.pileup(contig, start=start, stop=stop, truncate=True):
                yield {
                    "position": pileupcolumn.reference_pos,
                    "depth": pileupcolumn.get_num_aligned,
                }

        elif mode == "reads":
            for read in bam.fetch(contig, start=start, stop=stop):
                yield {
                    "id": read.query_name,
                    "contig": read.reference_name,
                    # Geometry
                    "start": read.reference_start,
                    "end": read.reference_end,
                    "is_reverse": read.is_reverse,
                    # CIGAR strings
                    "cigar": read.cigarstring,
                    "mapping_quality": read.mapping_quality,
                    # Mapping quality and flags
                    "query_name": read.query_name,
                    "is_unmapped": read.is_unmapped,
                }

        else:
            raise ValueError(f"Unknown BAM streaming mode: {mode}")

    def read_full(self, config: dict | None = None, **kwargs):
        """Full read into memory. Probably a bad idea for large BAMs"""
        # re-use preview logic to prevent server crashes
        io_config = {**(config or {}), **kwargs}
        io_config["strategy"] = "full"

        data, _ = self.preview(config=io_config, peek_limit=10_000_000)  # Arbitrary large number
        return data

    def preview(self, peek_limit: int, config: dict | None = None, **kwargs):
        """Safely peeks at the top N records or coverage points"""
        io_config = {**(config or {}), **kwargs}
        strategy = io_config.get("preview_strategy", "peek")

        data = []
        rows_read = 0
        eof_hit = True

        try:
            for record in self.stream(io_config):
                data.append(record)
                rows_read += 1

                if strategy == "peek" and rows_read >= peek_limit:
                    eof_hit = False
                    break

        except Exception as e:
            raise RuntimeError(f"Error while previewing BAM file '{self.path.name}': {e}")

        metadata = self.get_metadata()
        metadata.update({
            "io_strategy": f"Preview ({strategy})",
            "file_eof_hit": eof_hit,
            "preview_limit": peek_limit,
            "columns": list(data[0].keys()) if data else [],
        })
        return data, metadata
