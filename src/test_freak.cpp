TEST( Features2d_DescriptorExtractor_FREAK, regression )
{
    // TODO adjust the parameters below
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-freak",  (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
                                                 DescriptorExtractor::create("FREAK"), 0.010f );
    test.safe_run();
}