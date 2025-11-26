def quick_rate_test():
    
    API_KEY = 'ENTER_API_KEY' #Enter your Alphagenome API key here
    model = dna_client.create(API_KEY)
    
    print("QUICK RATE LIMIT TEST")
    print("=" * 30)
    
    # Test 10 quick queries
    start_time = time.time()
    successful_queries = 0
    
    for i in range(10):
        try:
            variant = genome.Variant(
                chromosome='chr12',
                position=109576074 + i*1000,  # Slightly different positions
                reference_bases='G',
                alternate_bases='A'
            )
            
            interval = genome.Interval(
                chromosome='chr12', 
                start=109576074 + i*1000 - 25000, 
                end=109576074 + i*1000 + 25000
            )
            
            outputs = model.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=['UBERON:0001157'],
                requested_outputs=[dna_client.OutputType.RNA_SEQ],
            )
            
            successful_queries += 1
            print(f"✅ Query {i+1}: Success")
            
        except Exception as e:
            print(f"Query {i+1}: Failed - {str(e)[:50]}...")
        
        time.sleep(0.1)  # Small delay
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate estimates
    avg_time_per_query = total_time / 10
    queries_per_hour = 3600 / avg_time_per_query
    genes_per_hour = queries_per_hour / 10  # Assuming 10 variants per gene
    
    print(f"\n QUICK RESULTS:")
    print(f"   - Successful queries: {successful_queries}/10")
    print(f"   - Average time per query: {avg_time_per_query:.2f}s")
    print(f"   - Estimated queries/hour: {queries_per_hour:.1f}")
    print(f"   - Estimated genes/hour: {genes_per_hour:.1f}")
    
    print(f"\n ANSWER:")
    print(f"   \"Approximately {int(genes_per_hour)} genes per hour\"")

# Run this first for a quick answer
quick_rate_test()
