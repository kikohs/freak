///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(FREAK, "Feature2D.FREAK",
    			  obj.info()->addParam(obj, "orientationNormalized", obj.orientationNormalized);
			      obj.info()->addParam(obj, "scaleNormalized", obj.scaleNormalized);
			      obj.info()->addParam(obj, "patternScale", obj.patternScale);
			      obj.info()->addParam(obj, "nbOctave", obj.nbOctave);
			      obj.info()->addParam(obj, "selectedPairs", obj.selectedPairs));

///////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////
   
static Algorithm* createFREAK() { return new FREAK(); }
static AlgorithmInfo& freak_info()
{
    static AlgorithmInfo freak_info_var("Feature2D.FREAK", createFREAK);
    return freak_info_var;
}

static AlgorithmInfo& freak_info_auto = freak_info();

AlgorithmInfo* FREAK::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        FREAK obj;
        freak_info().addParam(obj, "orientationNormalized", obj.orientationNormalized);
        freak_info().addParam(obj, "scaleNormalized", obj.scaleNormalized);
        freak_info().addParam(obj, "patternScale", obj.patternScale);
        freak_info().addParam(obj, "nbOctave", obj.nbOctave);
        freak_info().addParam(obj, "selectedPairs", obj.selectedPairs);
        
        initialized = true;
    }
    return &freak_info();
}
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////