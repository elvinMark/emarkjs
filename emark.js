////////////////////
/* Linear Algebra */
////////////////////

function init_vector(elems){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = 0;
	}
	var v = {
		n : elems,
		data : data
	};
	return v;
}

function init_matrix(rows,cols){
	var data = [];
	var aux;
	for(var i = 0;i<rows;i++){
		aux = init_vector(cols);
		data.push(aux);
	}
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}

function ones_vector(elems){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = 1;
	}
	var v = {
		n : elems,
		data : data
	};
	return v;
}

function ones_matrix(rows,cols){
	var data = [];
	var aux;
	for(var i = 0;i<rows;i++){
		aux = ones_vector(cols);
		data.push(aux);
	}
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}

function unit_vector(elems,pos){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = 0;
	}
	data[pos] = 1;
	var v = {
		n : elems,
		data : data
	};
	return v;	
}

function identity_matrix(dim){
	var data = [];
	var aux;
	for(var i = 0;i<dim;i++){
		aux = unit_vector(dim,i);
		data.push(aux);
	}
	var M = {
		rows : dim,
		cols : dim,
		data : data
	};
	return M;
}

function random_vector(elems){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = Math.random();
	}
	var v = {
		n : elems,
		data : data
	};
	return v;
}

function random_matrix(rows,cols){
	var data = [];
	var aux;
	for(var i = 0;i<rows;i++){
		aux = random_vector(cols);
		data.push(aux);
	}
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}

function str_2_vector(str,elems){
	var data;
	var aux;
	aux = str.split(" ");
	data = new Array(elems);
	for(var i =0;i<elems;i++){
		data[i] = parseFloat(aux[i]);
	}
	var v = {
		n : elems,
		data : data
	};
	return v;

}

function str_2_matrix(str,rows,cols){
	var data = [];
	var aux = str.split(";");
	for(var i = 0;i<rows;i++){
		data.push(str_2_vector(aux[i],cols));
	}
	console.log(data);
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}
// Functions to print in the console

function print_vector(v){
	var s = "";
	for(var i = 0;i<v.n;i++){
		s = s + v.data[i]+ " ";
	}
	console.log(s);
}

function print_matrix(M){
	var s = "";
	for(var i=0;i<M.rows;i++){
		for(var j=0;j<M.cols;j++){
			s = s + M.data[i].data[j] + " ";
		}
		s = s + "\n";
	}
	console.log(s);
}

// Basic operations with vector

function sum_vector(v1,v2){
	var out = init_vector(v1.n);
	for(var i = 0;i<v1.n;i++){
		out.data[i] = v1.data[i] + v2.data[i];
	}
	return out;
}

function diff_vector(v1,v2){
	var out = init_vector(v1.n);
	for(var i = 0;i<v1.n;i++){
		out.data[i] = v1.data[i] - v2.data[i];
	}
	return out;
}

function dot_vector(v1,v2){
	var out = 0;
	for(var i = 0;i<v1.n;i++){
		out = out + v1.data[i]*v2.data[i];
	}
	return out;
}

function prod_vector(l,v1){
	var out = init_vector(v1.n);
	for(var i = 0;i<v1.n;i++){
		out.data[i] = l*v1.data[i];
	}
	return out;	
}

function norm_vector(v1){
	var out=0;
	for(var i=0;i<v1.n;i++){
		out = out + v1.data[i]*v1.data[i]
	}
	return Math.sqrt(out);
}

// Basic Opeartions with matrix

function sum_matrix(m1,m2){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = m1.data[i].data[j]+m2.data[i].data[j];
		}
	}
	return out;
}

function diff_matrix(m1,m2){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = m1.data[i].data[j]-m2.data[i].data[j];
		}
	}
	return out;
}

function dot_matrix(m1,m2){
	var out = init_matrix(m1.rows,m2.cols);
	var s;
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m2.cols;j++){
			s = 0;
			for(var k =0;k<m1.cols;k++){
				s = s + m1.data[i].data[k]*m2.data[k].data[j]
			}
			out.data[i].data[j] = s;
		}
	}
	return out;
}

function prod_matrix(m1,m2){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = m1.data[i].data[j]*m2.data[i].data[j];
		}
	}
	return out;
}

function prod_num_matrix(l,m1){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = l*m1.data[i].data[j];
		}
	}	
	return out;
}

function trans_matrix(m1){
	var out = init_matrix(m1.cols,m1.rows);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[j].data[i] = m1.data[i].data[j];
		}
	}	
	return out;	
}

function lu_fact_matrix(m1){
	var L,U;
	var out = [];
	var s;
	L = identity_matrix(m1.rows,m1.cols);
	U = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j=i;j<m1.cols;j++){
			s = 0;
			for(var k=0;k<j;k++){
				s = s + L.data[i].data[k]*U.data[k].data[j];
			}
			U.data[i].data[j] = (m1.data[i].data[j] - s);
		}
		for(var j = i+1;j<m1.cols;j++){
			s = 0;
			for(var k = 0;k<i;k++){
				s = s + L.data[j].data[k]*U.data[k].data[i];
			}
			L.data[j].data[i] = (m1.data[j].data[i] - s)/U.data[i].data[i];
		}
	}
	out.push(L);
	out.push(U);
	return out;
}

function det_matrix(m1){
	var out = 1;
	var lu;
	lu = lu_fact_matrix(m1);
	for(var i = 0;i<m1.rows;i++){
		out = out * lu[1].data[i].data[i];
	}
	return out;
}

function inverse_matrix(m1){
	var out = init_matrix(m1.rows,m1.cols);
	var z = init_vector(m1.rows);
	var lu;
	var s;

	lu = lu_fact_matrix(m1);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.rows;j++){
			s = 0;
			for(var k = 0;k<j;k++){
				s = s + lu[0].data[j].data[k]*z.data[k]; 
			}
			if(i==j){
				z.data[j] = 1 - s;
			}
			else{
				z.data[j] = -s;
			} 
		}
		for(var j = m1.rows-1;j>=0;j--){
			s = 0;
			for(var k=j+1;k<m1.rows;k++){
				s = s + lu[1].data[j].data[k]*out.data[k].data[i];
			}
			out.data[j].data[i] = (z.data[j] - s)/lu[1].data[j].data[j];
		}
	}
	return out;
}

function qr_fact_matrix(m1){
	var Q,R;
	var out = [];
	var aux;
	var m = trans_matrix(m1);
	Q = init_matrix(m1.rows,m1.cols);
	R = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m.rows;i++){
		aux = init_vector(m.rows);
		for(var j = 0;j<i;j++){
			aux = sum_vector(aux,prod_vector(dot_vector(m.data[i],Q.data[j]),Q.data[j]));
		}
		aux = diff_vector(m.data[i],aux);
		R.data[i].data[i] = norm_vector(aux);
		aux = prod_vector(1/R.data[i].data[i],aux);
		Q.data[i] = aux;
		for(var j=0;j<i;j++){
			R.data[i].data[j] = dot_vector(m.data[i],Q.data[j]);
		}
	}
	out.push(trans_matrix(Q));
	out.push(trans_matrix(R));
	return out;
}

function eigen_matrix(m1,N){
	var qr;
	var out;
	out = m1;
	for(var i = 0;i<N;i++){
		qr  = qr_fact_matrix(out);
		out = dot_matrix(qr[1],qr[0]);
	}
	return out;
}

function solve_linear_system(m,b){
	var z = init_vector(b.n);
	var out = init_vector(b.n);
	var lu = lu_fact_matrix(m);
	var s;
	
	for(var j = 0;j<b.n;j++){
		s = 0;
		for(var k = 0;k<j;k++){
			s = s + lu[0].data[j].data[k]*z.data[k];
		}
		z.data[j] = b.data[j] - s;
	}
	for(var j = b.n-1;j>=0;j--){
		s = 0;
		for(var k = j+1;k<b.n;k++){
			s = s + lu[1].data[j].data[k]*out.data[k];
		}
		out.data[j] = (z.data[j] - s)/lu[1].data[j].data[j];
	}
	return out;
}

//////////////////
/*Neural Network*/
//////////////////

//Simple Multilayer Perceptron using sigmoid as activation function

function sigmoid_matrix(m1){
	var o;
	o = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			o.data[i].data[j] = 1.0/(1+Math.exp(-m1.data[i].data[j]));
		}
	}
	return o;
}

function init_layer(inputs,outputs){
	var l = {
		n_inputs : inputs,
		n_outputs : outputs,
		w : random_matrix(inputs,outputs),
		forward : function(inp){
			var o;
			o = sigmoid_matrix(dot_matrix(inp,this.w));
			this.i = inp;
			this.o = o;
		},
		print : function(){
			print_matrix(this.w);
		},
		backward : function(err){
			var ones = ones_matrix(err.rows,err.cols);
			var aux = prod_matrix(this.o,diff_matrix(ones,this.o));
			this.delta = prod_matrix(err,aux);
			return dot_matrix(this.delta,trans_matrix(this.w));
		},
		update : function(alpha){
			this.w = diff_matrix(this.w,prod_num_matrix(alpha,dot_matrix(trans_matrix(this.i),this.delta)));
		}
	};
	return l;
}

function init_mlp(structure){
	var l = [];
	for(var i = 0;i<structure.length-1;i++){
		l.push(init_layer(structure[i],structure[i+1]));
	}
	var n = {
		struct : structure,
		n_layers : structure.length-1,
		layers : l,
		forward : function(inp){
			var o;
			o = inp;
			for(var i=0;i<this.n_layers;i++){
				this.layers[i].forward(o);
				o = this.layers[i].o;
			}
			this.o = o;
		},
		print : function(){
			for(var i = 0;i<this.n_layers;i++){
				this.layers[i].print();
			}
		},
		backward : function(err){
			var aux = err;
			for(var i = this.n_layers-1;i>=0;i--){
				aux = this.layers[i].backward(aux);
			}
		},
		update : function(alpha){
			for(var i = 0;i<this.n_layers;i++){
				aux = this.layers[i].update(alpha);
			}	
		},
		train : function(in_set,out_set,alpha,N){
			var o;
			for(var i = 0;i<N;i++){
				this.forward(in_set);
				this.backward(diff_matrix(this.o,out_set));
				this.update(alpha);
			}
		}

	};
	return n;
}

/////////////////////////////
/*Solve non-linear equation*/
/////////////////////////////

function newton_method(fun_f,fun_df,x0,N){
	var x;
	x = x0;
	for(var i = 0;i < N;i++){
		x = x - fun_f(x)/fun_df(x0);
	}
	return x;
}

function bisection_method(fun_f,a,b,N){
	var xa;
	var xb;
	var mid;
	xa = a;
	xb = b;
	for(var i = 0;i<N;i++){
		mid = (xa + xb)/2;
		if(fun_f(xa)*fun_f(mid) <0){
			xb = mid;
		}
		else{
			xa = mid;
		}
	}
	return xa;
}

//////////////
/* Calculus */
//////////////

function derivate_function(fun_f,x0,h){
	return (fun_f(x0+h) - fun_f(x0))/h;
}

function integral_function(fun_f,a,b,N){
	var h = (b-a)/N;
	var s = 0;
	var x = a;
	for(var i = 0;i<N;i++){
		s = s + fun_f(x)*h;
		x = x + h;
	}
	return h;
}

////////////////
/* Statistics */
////////////////

function average(v){
	var out;
	out = 0;
	for(var i =0;i<v.n;i++){
		out = out + v.data[i];
	}
	out = out/v.n;
	return out;
}

function std_deviation(v){
	var avg = average(v);
	var out = 0;
	for(var i = 0;i < v.n;i++){
		out = out + (v.data[i] - avg)*(v.data[i] - avg);
	}
	out = Math.sqrt(out/v.n);
	return out;
}

function combinatory(n,k){
	if(k==0 || k == n)
		return 1;
	return combinatory(n-1,k) + combinatory(n-1,k-1);
}

function factorial(n){
	if(n==0 || n==1)
		return 1;
	return n*factorial(n-1);
}

function linear_regression(x_data,y_data){
	var A = init_matrix(2,2);
	var b = init_vector(2);
	var sx = 0;
	var sy = 0;
	for(var i =0;i<x_data.n;i++){
		sx = sx + x_data.data[i];
	}
	for(var i =0;i<y_data.n;i++){
		sy = sy + y_data.data[i];
	}
	A.data[0].data[0] = dot_vector(x_data,x_data);
	A.data[0].data[1] = sx;
	A.data[1].data[0] = sx;
	A.data[1].data[1] = x_data.n;
	b.data[0] = dot_vector(x_data,y_data);
	b.data[1] = sy;
	print_matrix(A);
	print_vector(b);
	return solve_linear_system(A,b);
}

function gaussian_distribution(x,avg,std){
	return (1/Math.sqrt(2*Math.PI*std*std))*Math.exp(-Math.pow(x-avg,2)/(2*Math.pow(std,2)));
}

function binomial_distribution(x,p,N){
	return combinatory(N,x)*Math.pow(p,x)*Math.pow(1-p,N-x);
}