function convObj = convBlock(fh,fw,fc,k)
    convObj = dagnn.Conv('size', [fh fw fc k], 'hasBias', true, 'pad', 1);
end