

__kernel void findHashes(__global char *d_css, __global uint *d_iss,const int d_len , const int pattern_length,const int d, const int p, const int q)
{
	//in each thread
	int i = 0;
	int ind = d_len * get_global_id(0);


	int pw = 1;
	int t=0;
	//initial hash value
	for (; i < pattern_length; i++) {
		
		t=(d * t + d_css[ind+i]) % q;
	}
	d_iss[ind]=t;
	//next hash value 
	for (i = 1; i < d_len - pattern_length + 1; i++) {
		
        t=(d * (d_iss[ind+i - 1] - (d_css[ind+i - 1]) * p) + d_css[ind+i + pattern_length - 1]) ;
		t=t%q;

		if (t < 0)
           t = (t + q);
		d_iss[ind+i]=t;

	}

}

__kernel void seekPattern(__global char *d_css, __global uint *d_iss, const uint d_len,const uint pattern_length,__global char *pattern,const uint p0 /*,__global int *result*/) 
{
	int i = 0;
        int j=0;
		int id=get_global_id(0);
	int ind = d_len * id;
	//find the match of pattern in the text using hash table.
	for (i = 0; i < d_len - pattern_length + 1; i++) {
		if (d_iss[ind+i] == p0) {
			for (j = 0; j < pattern_length; j++) {
				if (pattern[j] != d_css[ind+i + j]) {
					//result[ind+i]=-	1;
					break;
				} else if (j == pattern_length - 1) {
				
				//result[ind+i]=id*(d_len-pattern_length+1)+i-pattern_length+1;
				printf("pos:%d ",id*(d_len-pattern_length+1)+i-pattern_length+1);
				}
			}
		}
	}

}