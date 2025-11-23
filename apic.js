let main = async () => {
  await ti.init();

  let quality = 1;
  //let n_particles = 9000 * quality ** 2;
  //let n_grid = 128 * quality;
  //let dx = 1 / n_grid;

    const res_level = 1;
    const n_particles = 8192 * 1;

// We use a simple object to mimic the vector structure
    const n_grid = { 
    x: 48 * res_level * 1 , 
    y: 48 * res_level 
    };

const dx = 1 / n_grid.y;
const domain_width = n_grid.x * dx;
const domain_height = n_grid.y * dx;
const dt = 2e-4 / res_level;

  let inv_dx = n_grid;
  let dt = 1e-4 / quality;
  let p_vol = (dx * 0.5) ** 2;
  let p_rho = 1;
  let p_mass = p_vol * p_rho;
  let E = 5e3; // Young's modulus a
  let nu = 0.2; // Poisson's ratio
  let mu_0 = E / (2 * (1 + nu));
  let lambda_0 = (E * nu) / ((1 + nu) * (1 - 2 * nu)); // Lame parameters
  let x = ti.Vector.field(2, ti.f32, [n_particles]); // position
  let v = ti.Vector.field(2, ti.f32, [n_particles]); // velocity
  let C = ti.Matrix.field(2, 2, ti.f32, [n_particles]); // affine vel field
  let F = ti.Matrix.field(2, 2, ti.f32, n_particles); // deformation gradient
  let material = ti.field(ti.i32, [n_particles]); // material id
  let Jp = ti.field(ti.f32, [n_particles]); // plastic deformation
  let grid_v = ti.Vector.field(2, ti.f32, [n_grid, n_grid]);
  let grid_m = ti.field(ti.f32, [n_grid, n_grid]);

  let n_nodes = 10;
  let ground_x_values = [...Array(n_nodes).keys()];
  let ground_y_values = ti.field(ti.f32, n_nodes);

  //from slider values:
  let x_val = 1;

  let para_a = 1;
  const para_a_slider_elm = document.getElementById("para_a");

  let para_b = 0.4;
  let kick_b = 0.2;
  let kick_h = 0.1;
  let kick_a = -10;

  let inflow = 10;
  let river_depth = 0.5;

  const bound = 3;

  const img_size = 512;
  let image = ti.Vector.field(4, ti.f32, [img_size, img_size]);
  const group_size = n_particles / 3;

  let riverbed = (x_val, para_a, para_b, kick_b, kick_h, kick_a) => {
    let y = 0.0;
    let normal = [0.0, 1.0];
    let parabola_c = 3 * dx;
    let ground_transition_x = 0.7;

    let kicker_c = para_a * (kick_b - para_b) ** 2 + 3 * dx + kick_h;
    let A = kick_a - para_a;
    let B = 2 * (para_a * para_b - kick_a * kick_b);
    let D = kick_a * kick_b ** 2 - para_a * para_b ** 2 + kicker_c - parabola_c;

    // Solve quadratic intersection
    let delta = B ** 2 - 4 * A * D;
    // Determine kicker range. Using dummy values if delta < 0 to avoid NaN
    let kicker_start = 0.0;
    let kicker_end = 0.0;

    if (delta >= 0) {
      kicker_start = (-B + ti.sqrt(delta)) / (2 * A);
      kicker_end = (-B - ti.sqrt(delta)) / (2 * A);
    }

    let ground_transition_y =
      para_a * (ground_transition_x - para_b) ** 2 + 3 * dx;

    if (x_val > kicker_start && x_val < kicker_end) {
      y = kick_a * (x_val - kick_b) ** 2 + kicker_c;
      let dy_dx = 2 * kick_a * (x_val - kick_b);
      let norm_raw = [-dy_dx, 1.0];
      normal = ti.normalized(norm_raw);
    } else if (x_val < ground_transition_x) {
      y = para_a * (x_val - para_b) ** 2 + parabola_c;
      let dy_dx = 2 * para_a * (x_val - para_b);
      let norm_raw = [-dy_dx, 1.0];
      normal = ti.normalized(norm_raw);
    } else {
      y = ground_transition_y;
      normal = [0.0, 1.0];
    }

    return { y, normal };
  };

  ti.addToKernelScope({
    n_particles,
    n_grid,
    dx,
    inv_dx,
    dt,
    p_vol,
    p_rho,
    p_mass,
    E,
    nu,
    mu_0,
    lambda_0,
    x,
    v,
    C,
    F,
    material,
    Jp,
    grid_v,
    grid_m,
    image,
    img_size,
    group_size,
    bound,
    riverbed,
    para_a,
    para_b,
    kick_a,
    kick_b,
    kick_h,
    inflow,
    river_depth,
  });

  let substep = ti.kernel({ f: ti.template() }, (para_a_slider, f) => {
    let test = f[0];
    for (let I of ti.ndrange(n_grid, n_grid)) {
      grid_v[I] = [0, 0];
      grid_m[I] = 0;
    }
    for (let p of ti.range(n_particles)) {
      let base = i32(x[p] * inv_dx - 0.5);
      let fx = x[p] * inv_dx - f32(base);
      let w = [
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1) ** 2,
        0.5 * (fx - 0.5) ** 2,
      ];
      F[p] = (
        [
          [1.0, 0.0],
          [0.0, 1.0],
        ] +
        dt * C[p]
      ).matmul(F[p]);

      let h = f32(max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p])))));
      if (material[p] == 1) {
        h = 0.3;
      }
      let mu = mu_0 * h;
      let la = lambda_0 * h;
      if (material[p] == 0) {
        mu = 0.0;
      }
      let svd = ti.svd2D(F[p]);
      let U = svd.U;
      let sig = svd.E;
      let V = svd.V;
      let J = 1.0;
      for (let d of ti.static(ti.range(2))) {
        let new_sig = sig[[d, d]];
        if (material[p] == 2) {
          // Plasticity
          new_sig = min(max(sig[[d, d]], 1 - 2.5e-2), 1 + 4.5e-3);
        }
        Jp[p] = (Jp[p] * sig[[d, d]]) / new_sig;
        sig[[d, d]] = new_sig;
        J = J * new_sig;
      }
      if (material[p] == 0) {
        F[p] =
          [
            [1.0, 0.0],
            [0.0, 1.0],
          ] * sqrt(J);
      } else if (material[p] == 2) {
        F[p] = U.matmul(sig).matmul(V.transpose());
      }
      let stress =
        (2 * mu * (F[p] - U.matmul(V.transpose()))).matmul(F[p].transpose()) +
        [
          [1.0, 0.0],
          [0.0, 1.0],
        ] *
          la *
          J *
          (J - 1);
      stress = -dt * p_vol * 4 * inv_dx * inv_dx * stress;
      let affine = stress + p_mass * C[p];

      for (let i of ti.static(ti.range(3))) {
        for (let j of ti.static(ti.range(3))) {
          let offset = [i, j];
          let dpos = (f32(offset) - fx) * dx;
          let weight = w[[i, 0]] * w[[j, 1]];
          grid_v[base + offset] +=
            weight * (p_mass * v[p] + affine.matmul(dpos));
          grid_m[base + offset] += weight * p_mass;
        }
      }
    }
    for (let I of ndrange(n_grid, n_grid)) {
      let i = I[0];
      let j = I[1];
      if (grid_m[I] > 0) {
        grid_v[I] = (1 / grid_m[I]) * grid_v[I];
        grid_v[I][1] -= dt * 50;

        if (i < 3 && grid_v[I][0] < 0) {
          //grid_v[I][0] = 0;
        }
        if (i > n_grid - 3 && grid_v[I][0] > 0) {
          //grid_v[I][0] = 0;
        }
        if (j < 3 && grid_v[I][1] < 0) {
          grid_v[I][1] = 0;
        }
        if (j > n_grid - 3 && grid_v[I][1] > 0) {
          grid_v[I][1] = 0;
        }

        //Boundary conditions

        // Inflow
        if (i < bound && grid_v[I][0] < 0) {
          grid_v[I][0] = inflow;
        }
        // Outflow
        if (i > n_grid - bound && grid_v[I][0] > 0) {
          grid_v[I][0] = inflow;
        }
        // riverbed
        let xi = i * dx;
        let rb = riverbed(xi, para_a_slider, para_b, kick_b, kick_h, kick_a);

        let y_bound = rb.y; // 0.5;
        let normal = rb.normal; // [0, 1];

        let y_j = ti.i32(y_bound * n_grid - 0.5) + 1;

        let normal_component = grid_v[I].dot(normal);

        if (j <= y_j && normal_component < 0) {
          grid_v[I] -= normal_component * normal;
        }

        // Ceiling (prevent flying too high)
        if (j > n_grid - bound && grid_v[I].y > 0) {
          grid_v[I].y = 0;
        }
      }
    }
    for (let p of range(n_particles)) {
      let base = i32(x[p] * inv_dx - 0.5);
      let fx = x[p] * inv_dx - f32(base);
      let w = [
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
      ];
      let new_v = [0.0, 0.0];
      let new_C = [
        [0.0, 0.0],
        [0.0, 0.0],
      ];
      for (let i of ti.static(ti.range(3))) {
        for (let j of ti.static(ti.range(3))) {
          let dpos = f32([i, j]) - fx;
          let g_v = grid_v[base + [i, j]];
          let weight = w[[i, 0]] * w[[j, 1]];
          new_v = new_v + weight * g_v;
          new_C = new_C + 4 * inv_dx * weight * g_v.outerProduct(dpos);
        }
      }
      v[p] = new_v;
      C[p] = new_C;
      x[p] = x[p] + dt * new_v;

      // Respawn logic

      let rb = riverbed(x[p][0], para_a_slider, para_b, kick_b, kick_h, kick_a);
      let y_limit = rb.y;
      let normal = rb.normal;

      if (
        x[p][0] > 1.0 - 3 * dx ||
        x[p][0] < dx ||
        x[p][1] < y_limit ||
        x[p][1] > 1.0 - 3 * dx
      ) {
        x[p] = [ti.random() * 3 * dx + dx, ti.random() * river_depth];
        x[p] += [0.0, y_limit];
        v[p] = [normal.y * 2, -normal.x];
        //J[p] = 1.0;
        //C[p] = [[0.0, 0.0], [0.0, 0.0]];
      }
    }
  });

  let reset = ti.kernel(() => {
    for (let i of range(n_particles)) {
      let group_id = i32(ti.floor(i / group_size));

      x[i] = [ti.random() * 0.2 + 0.3, ti.random() * 0.2 + 0.4];
      material[i] = 0;
      //f[i] = 0 ;
      v[i] = [0, 0];
      F[i] = [
        [1, 0],
        [0, 1],
      ];
      Jp[i] = 1;
      C[i] = [
        [0, 0],
        [0, 0],
      ];
    }
  });

  let render = ti.kernel(() => {
    for (let I of ndrange(img_size, img_size)) {
      image[I] = [0.067, 0.184, 0.255, 1.0];
    }
    for (let i of range(n_particles)) {
      let pos = x[i];
      let ipos = i32(pos * img_size);
      let this_color = f32([0, 0, 0, 0]);
      if (material[i] == 0) {
        this_color = [0, 0.5, 0.5, 1.0];
      } else if (material[i] == 1) {
        this_color = [0.93, 0.33, 0.23, 1.0];
      } else if (material[i] == 2) {
        this_color = [1, 1, 1, 1.0];
      }
      image[ipos] = this_color;
    }
  });

  const getCanvasNormalizedXY = (event) => {
    var rect = htmlCanvas.getBoundingClientRect();
    if (event.touches) {
      return {
        x: (event.touches[0].clientX - rect.left) / rect.width,
        y: 1.0 - (event.touches[0].clientY - rect.top) / rect.height,
      };
    } else {
      return {
        x: (event.clientX - rect.left) / rect.width,
        y: 1.0 - (event.clientY - rect.top) / rect.height,
      };
    }
  };

  const mouseMoveListener = (event) => {
    canvasCoords = getCanvasNormalizedXY(event);
    xi = Math.floor(canvasCoords.x / n_nodes);
    if ((xi >= 0) & (xi < n_nodes)) {
      ground_y_values.set([xi], canvasCoords.y);
    }
    console.log(canvasCoords);
  };

  // document.addEventListener("mousedown", mouseDownListener);
  document.addEventListener("mousemove", mouseMoveListener);
  // document.addEventListener("mouseup", mouseupListener);

  // document.addEventListener("touchstart", mouseDownListener);
  document.addEventListener("touchmove", mouseMoveListener);
  // document.addEventListener("touchend", mouseupListener);

  const htmlCanvas = document.getElementById("result_canvas");
  htmlCanvas.width = img_size;
  htmlCanvas.height = img_size;
  const canvas = new ti.Canvas(htmlCanvas);

  reset();

  let i = 0;
  async function frame() {
    if (window.shouldStop) {
      return;
    }

    const a = parseFloat(para_a_slider_elm.value);
    //f.from_array
    for (let i = 0; i < Math.floor(2e-3 / dt); ++i) {
      substep(a, ground_y_values);
    }

    render();

    i = i + 1;
    canvas.setImage(image);
    requestAnimationFrame(frame);
  }
  await frame();
};
// This is just because StackBlitz has some weird handling of external scripts.
// Normally, you would just use `<script src="https://unpkg.com/taichi.js/dist/taichi.umd.js"></script>` in the HTML
const script = document.createElement("script");
script.addEventListener("load", function () {
  main();
});

script.src = "https://unpkg.com/taichi.js/dist/taichi.umd.js";
// Append to the `head` element
document.head.appendChild(script);
