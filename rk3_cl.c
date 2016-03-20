{{header}}


    #include <pyopencl-complex.h>


    {{complex_type}}_t foo_add({{complex_type}}_t a, {{complex_type}}_t b) {
        return a+b;
    }

    {{complex_type}}_t foo_sub({{complex_type}}_t a, {{complex_type}}_t b) {
        return a-b;
    }

    {{complex_type}}_t foo_mul({{complex_type}}_t a, {{complex_type}}_t b) {
        return {{complex_type}}_new(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
    }

    __kernel
    void runge_kutta(
        __global const {{complex_type}}_t *h, __global {{complex_type}}_t *p,
       // __global const int *row_data, __global const int *col_data,
       // __global const {{real_type}} *Aw, __global const {{real_type}} *As, __global const {{real_type}} *ww, __global const {{real_type}} *ws,
        int t
    )
    {
        int row = get_local_id(1);
        int col = get_local_id(0);

        int mat_id = get_group_id(0) * get_num_groups(0) + get_group_id(1);


        __local {{complex_type}}_t k1[{{mat_size}}];
        __local {{complex_type}}_t k2[{{mat_size}}];
        __local {{complex_type}}_t k3[{{mat_size}}];


        {{complex_type}}_t imag = {{complex_type}}_new(0, 1);


        {{complex_type}}_t entry = 0;
        for (int e = 0; e < {{mat_dim}}; ++e) {
            entry += h[mat_id*{{mat_size}} + row * {{mat_dim}} + e] * p[mat_id*{{mat_size}} + e * {{mat_dim}} + col];
            entry -= p[mat_id*{{mat_size}} + row * {{mat_dim}} + e] * h[mat_id*{{mat_size}} + e * {{mat_dim}} + col];
        }
        k1[mat_id*{{mat_size}} + row * {{mat_dim}} + col] = {{complex_type}}_mul(-imag, entry);

        barrier(CLK_LOCAL_MEM_FENCE);

        entry = 0;
        for (int e = 0; e < {{mat_dim}}; ++e) {
            entry += foo_mul(
                h[mat_id*{{mat_size}} + row * {{mat_dim}} + e],
                foo_add(
                    p[mat_id*{{mat_size}} + e * {{mat_dim}} + col],
                    0.5*{{step_size}}*k1[mat_id*{{mat_size}} + e * {{mat_dim}} + col]
                )
            );

            entry -= foo_mul(
                foo_add(
                    p[mat_id*{{mat_size}} + row * {{mat_dim}} + e],
                    0.5*{{step_size}}*k1[mat_id*{{mat_size}} + row * {{mat_dim}} + e]
                ),
                h[mat_id*{{mat_size}} + e * {{mat_dim}} + col]
            );
        }
        k2[mat_id*{{mat_size}} + row * {{mat_dim}} + col] = foo_mul(-imag, entry);

        barrier(CLK_LOCAL_MEM_FENCE);


        entry = 0;
        for (int e = 0; e < {{mat_dim}}; ++e) {
            entry += foo_mul(
                h[mat_id*{{mat_size}} + row * {{mat_dim}} + e],
                foo_add(
                    p[mat_id*{{mat_size}} + e * {{mat_dim}} + col],
                    foo_sub(
                        2*{{step_size}}*k2[mat_id*{{mat_size}} + e * {{mat_dim}} + col],
                        {{step_size}}*k1[mat_id*{{mat_size}} + e * {{mat_dim}} + col]
                    )
                )
            );

            entry -= foo_mul(
                foo_add(
                    p[mat_id*{{mat_size}} + row * {{mat_dim}} + e],
                    foo_sub(
                        2*{{step_size}}*k2[mat_id*{{mat_size}} + row * {{mat_dim}} + e],
                        {{step_size}}*k1[mat_id*{{mat_size}} + row * {{mat_dim}} + e]
                    )
                ),
                h[mat_id*{{mat_size}} + e * {{mat_dim}} + col]
            );
        }
        k3[mat_id*{{mat_size}} + row * {{mat_dim}} + col] = foo_mul(-imag, entry);

        barrier(CLK_LOCAL_MEM_FENCE);

        p[mat_id*{{mat_size}} + row * {{mat_dim}} + col] = foo_add(
            p[mat_id*{{mat_size}} + row * {{mat_dim}} + col],
            {{step_size}} * foo_add(
                {{b1}}*k1[mat_id*{{mat_size}} + row * {{mat_dim}} + col],
                {{complex_type}}_add(
                    {{b2}}*k2[mat_id*{{mat_size}} + row * {{mat_dim}} + col],
                    {{b1}}*k3[mat_id*{{mat_size}} + row * {{mat_dim}} + col]
                )
            )
        );
    }