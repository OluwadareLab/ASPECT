import pybedtools

def extract_sequence(chrm, start, end, fasta_file, strand):
    bed_interval = pybedtools.BedTool(f"{chrm}\t{start}\t{end}", from_string=True)
    sequence = bed_interval.sequence(fi=fasta_file, s=True, )
    return sequence

if __name__ == "__main__":
    chrm = "chr2"
    start = 196780488
    end = 196780628
    fasta_file = "hg38.fa"
    strand = "-"

    sequence = extract_sequence(chrm, start, end, fasta_file, strand)
    print(open(sequence.seqfn).read())
