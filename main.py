import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Análise Inercial da Marcha",
    layout="wide"
)

st.title("Análise de Acelerômetro e Giroscópio")

FS = 100


# ============================================================
# FUNÇÕES
# ============================================================

def michaelis_menten(x, DC, Vmax, n, Km):
    return DC + Vmax * (x ** n) / ((x ** n) + (Km ** n))


def butterworth_filter(data, cutoff, fs, order=4, btype="low"):

    nyquist = 0.5 * fs

    if btype == "bandpass":

        if cutoff[0] <= 0:
            raise ValueError(
                "A frequência inferior deve ser maior que 0 Hz."
            )

        if cutoff[1] >= nyquist:
            raise ValueError(
                f"A frequência superior deve ser menor que {nyquist:.1f} Hz."
            )

        if cutoff[0] >= cutoff[1]:
            raise ValueError(
                "A frequência inferior deve ser menor que a superior."
            )

        normal_cutoff = [
            cutoff[0] / nyquist,
            cutoff[1] / nyquist
        ]

    else:

        if cutoff <= 0 or cutoff >= nyquist:
            raise ValueError(
                f"A frequência de corte deve estar entre 0 e {nyquist:.1f} Hz."
            )

        normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype=btype
    )

    return filtfilt(
        b,
        a,
        data
    )


def preprocess_sensor(
    df,
    fs=100,
    normalize_gravity=False
):

    """
    Espera:
    coluna 0 = tempo em ms
    coluna 1 = X
    coluna 2 = Y
    coluna 3 = Z
    """

    time_raw = pd.to_numeric(
        df.iloc[:, 0],
        errors="coerce"
    ).to_numpy(dtype=float)

    x_raw = pd.to_numeric(
        df.iloc[:, 1],
        errors="coerce"
    ).to_numpy(dtype=float)

    y_raw = pd.to_numeric(
        df.iloc[:, 2],
        errors="coerce"
    ).to_numpy(dtype=float)

    z_raw = pd.to_numeric(
        df.iloc[:, 3],
        errors="coerce"
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(time_raw)
        & np.isfinite(x_raw)
        & np.isfinite(y_raw)
        & np.isfinite(z_raw)
    )

    time_raw = time_raw[valid]
    x_raw = x_raw[valid]
    y_raw = y_raw[valid]
    z_raw = z_raw[valid]

    if len(time_raw) < 10:
        raise ValueError(
            "Número insuficiente de amostras."
        )

    sort_idx = np.argsort(
        time_raw
    )

    time_raw = time_raw[
        sort_idx
    ]

    x_raw = x_raw[
        sort_idx
    ]

    y_raw = y_raw[
        sort_idx
    ]

    z_raw = z_raw[
        sort_idx
    ]

    time_unique, unique_idx = np.unique(
        time_raw,
        return_index=True
    )

    time_raw = time_unique

    x_raw = x_raw[
        unique_idx
    ]

    y_raw = y_raw[
        unique_idx
    ]

    z_raw = z_raw[
        unique_idx
    ]

    # --------------------------------------------------------
    # NORMALIZAÇÃO PARA g
    # --------------------------------------------------------

    if normalize_gravity:

        max_abs = np.max(
            np.abs(
                np.concatenate(
                    [
                        x_raw,
                        y_raw,
                        z_raw
                    ]
                )
            )
        )

        if max_abs > 9:

            x_raw = (
                x_raw
            )

            y_raw = (
                y_raw
            )

            z_raw = (
                z_raw
            )

    # --------------------------------------------------------
    # DETREND
    # --------------------------------------------------------

    x_detrended = signal.detrend(
        x_raw
    )

    y_detrended = signal.detrend(
        y_raw
    )

    z_detrended = signal.detrend(
        z_raw
    )

    # --------------------------------------------------------
    # INTERPOLAÇÃO
    # --------------------------------------------------------

    step_ms = (
        1000 / fs
    )

    time_interp = np.arange(
        start=time_raw[0],
        stop=time_raw[-1],
        step=step_ms
    )

    interp_x = scipy.interpolate.interp1d(
        time_raw,
        x_detrended,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    interp_y = scipy.interpolate.interp1d(
        time_raw,
        y_detrended,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    interp_z = scipy.interpolate.interp1d(
        time_raw,
        z_detrended,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    x = interp_x(
        time_interp
    )

    y = interp_y(
        time_interp
    )

    z = interp_z(
        time_interp
    )

    t = (
        time_interp / 1000
    )

    norm = np.sqrt(
        x ** 2
        + y ** 2
        + z ** 2
    )

    return (
        t,
        x,
        y,
        z,
        norm
    )


def first_index_above(
    data,
    threshold
):

    indices = np.where(
        data >= threshold
    )[0]

    if len(indices) == 0:

        raise ValueError(
            "Não foi encontrado cruzamento do limiar."
        )

    return indices[
        0
    ]


def first_index_below(
    data,
    threshold
):

    indices = np.where(
        data <= threshold
    )[0]

    if len(indices) == 0:

        raise ValueError(
            "Não foi encontrado cruzamento do limiar."
        )

    return indices[
        0
    ]


# ============================================================
# ABAS
# ============================================================

tab_acc, tab_gyro = st.tabs(
    [
        "Acelerômetro",
        "Giroscópio"
    ]
)


# ============================================================
# ACELERÔMETRO
# ============================================================

with tab_acc:

    st.header(
        "Acelerômetro"
    )

    uploaded_acc = st.file_uploader(
        "Carregue o arquivo do acelerômetro",
        type=[
            "txt",
            "csv"
        ],
        key="accelerometer"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        acc_cutoff = st.number_input(
            "Filtro passa-baixa do acelerômetro (Hz)",
            min_value=0.1,
            max_value=20.0,
            value=0.5,
            step=0.1
        )

    with col2:

        use_hyperbolic_model = st.checkbox(
            "Aplicar modelos hiperbólicos",
            value=True
        )

    if uploaded_acc is not None:

        try:

            df_acc = pd.read_csv(
                uploaded_acc,
                sep=";"
            )

            if df_acc.shape[
                1
            ] < 4:

                st.error(
                    "O arquivo precisa possuir pelo menos "
                    "4 colunas: tempo, X, Y e Z."
                )

                st.stop()

            (
                t_acc,
                acc_x,
                acc_y,
                acc_z,
                acc_norm
            ) = preprocess_sensor(
                df_acc,
                fs=FS,
                normalize_gravity=True
            )

            acc_x_filtered = butterworth_filter(
                acc_x,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            acc_y_filtered = butterworth_filter(
                acc_y,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            acc_z_filtered = butterworth_filter(
                acc_z,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            acc_norm_filtered = butterworth_filter(
                acc_norm,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            st.session_state[
                "t_acc"
            ] = t_acc

            st.session_state[
                "acc_norm_filtered"
            ] = acc_norm_filtered

            # ----------------------------------------------------
            # GRÁFICO
            # ----------------------------------------------------

            st.subheader(
                "Norma da aceleração"
            )

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                t_acc,
                acc_norm_filtered,
                linewidth=1.3
            )

            ax.set_xlabel(
                "Tempo (s)"
            )

            ax.set_ylabel(
                "Aceleração (g)"
            )

            ax.set_title(
                "Norma da aceleração"
            )

            ax.grid(
                alpha=0.3
            )

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )

            # ====================================================
            # MODELOS
            # ====================================================

            if use_hyperbolic_model:

                st.subheader(
                    "Modelos hiperbólicos"
                )

                lim1 = 100

                if len(
                    acc_norm_filtered
                ) <= (
                    2 * lim1 + 20
                ):

                    st.error(
                        "O registro é muito curto para "
                        "aplicação do modelo."
                    )

                else:

                    try:

                        peak_index = np.argmax(
                            acc_norm_filtered
                        )

                        if (
                            peak_index <= lim1
                            or
                            peak_index >=
                            len(
                                t_acc
                            ) - lim1
                        ):

                            raise ValueError(
                                "O pico principal está muito próximo "
                                "das extremidades."
                            )

                        # =========================================
                        # MODELO FINAL
                        # =========================================

                        x_data_end = (
                            t_acc[
                                peak_index:
                                len(t_acc) - lim1
                            ]
                            -
                            t_acc[
                                peak_index
                            ]
                        )

                        y_data_end = (
                            acc_norm_filtered[
                                peak_index:
                                len(t_acc) - lim1
                            ]
                        )

                        y_data_end_reverse = (
                            y_data_end[
                                ::-1
                            ]
                        )

                        initial_guess = [
                            0.1,
                            0.3,
                            9,
                            3
                        ]

                        params_end, _ = curve_fit(
                            michaelis_menten,
                            x_data_end,
                            y_data_end_reverse,
                            p0=initial_guess,
                            maxfev=50000
                        )

                        (
                            DC_fit_end,
                            Vmax_fit_end,
                            n_fit_end,
                            Km_fit_end
                        ) = params_end

                        y_fit_end = michaelis_menten(
                            x_data_end,
                            DC_fit_end,
                            Vmax_fit_end,
                            n_fit_end,
                            Km_fit_end
                        )

                        y_fit_end = y_fit_end[
                            ::-1
                        ]

                        x_data_end_absolute = (
                            x_data_end
                            +
                            t_acc[
                                peak_index
                            ]
                        )

                        # =========================================
                        # MODELO INICIAL
                        # =========================================

                        x_data_start = (
                            t_acc[
                                lim1:
                                peak_index
                            ]
                            -
                            t_acc[
                                lim1
                            ]
                        )

                        y_data_start = (
                            acc_norm_filtered[
                                lim1:
                                peak_index
                            ]
                        )

                        params_start, _ = curve_fit(
                            michaelis_menten,
                            x_data_start,
                            y_data_start,
                            p0=initial_guess,
                            maxfev=50000
                        )

                        (
                            DC_fit_start,
                            Vmax_fit_start,
                            n_fit_start,
                            Km_fit_start
                        ) = params_start

                        y_fit_start = michaelis_menten(
                            x_data_start,
                            DC_fit_start,
                            Vmax_fit_start,
                            n_fit_start,
                            Km_fit_start
                        )

                        x_data_start_absolute = (
                            x_data_start
                            +
                            t_acc[
                                lim1
                            ]
                        )

                        # =========================================
                        # LIMIARES
                        # =========================================

                        max_start = np.max(
                            y_fit_start
                        )

                        max_end = np.max(
                            y_fit_end
                        )

                        start_25 = first_index_above(
                            y_fit_start,
                            max_start * 0.25
                        )

                        start_50 = first_index_above(
                            y_fit_start,
                            max_start * 0.50
                        )

                        start_75 = first_index_above(
                            y_fit_start,
                            max_start * 0.75
                        )

                        end_75 = first_index_below(
                            y_fit_end,
                            max_end * 0.75
                        )

                        end_50 = first_index_below(
                            y_fit_end,
                            max_end * 0.50
                        )

                        end_25 = first_index_below(
                            y_fit_end,
                            max_end * 0.25
                        )

                        # =========================================
                        # INÍCIO E FINAL
                        # =========================================

                        inicio_atividade = (
                            x_data_start_absolute[
                                start_25
                            ]
                        )

                        fim_atividade = (
                            x_data_end_absolute[
                                end_25
                            ]
                        )

                        st.session_state[
                            "inicio_atividade"
                        ] = inicio_atividade

                        st.session_state[
                            "fim_atividade"
                        ] = fim_atividade

                        # =========================================
                        # GRÁFICO
                        # =========================================

                        fig, ax = plt.subplots(
                            figsize=(12, 5)
                        )

                        ax.plot(
                            t_acc,
                            acc_norm_filtered,
                            linewidth=1.3,
                            label="Norma"
                        )

                        ax.plot(
                            x_data_start_absolute,
                            y_fit_start,
                            linewidth=2,
                            label="Modelo inicial"
                        )

                        ax.plot(
                            x_data_end_absolute,
                            y_fit_end,
                            linewidth=2,
                            label="Modelo final"
                        )

                        ax.axvline(
                            inicio_atividade,
                            linestyle="--",
                            label="Início"
                        )

                        ax.axvline(
                            fim_atividade,
                            linestyle="--",
                            label="Final"
                        )

                        ax.axvspan(
                            inicio_atividade,
                            fim_atividade,
                            alpha=0.08
                        )

                        ax.set_xlabel(
                            "Tempo (s)"
                        )

                        ax.set_ylabel(
                            "Aceleração (g)"
                        )

                        ax.set_title(
                            "Modelagem da atividade"
                        )

                        ax.grid(
                            alpha=0.3
                        )

                        ax.legend()

                        st.pyplot(
                            fig
                        )

                        plt.close(
                            fig
                        )

                        # =========================================
                        # CÁLCULOS
                        # =========================================

                        amplitude_maxima_aceleracao = (
                            Vmax_fit_start
                        )

                        amplitude_maxima_desaceleracao = (
                            Vmax_fit_end
                        )

                        tempo_aceleracao = (
                            x_data_start_absolute[
                                start_75
                            ]
                            -
                            x_data_start_absolute[
                                start_25
                            ]
                        )

                        tempo_desaceleracao = (
                            x_data_end_absolute[
                                end_25
                            ]
                            -
                            x_data_end_absolute[
                                end_75
                            ]
                        )

                        tempo_constante = (
                            x_data_end_absolute[
                                end_75
                            ]
                            -
                            x_data_start_absolute[
                                start_75
                            ]
                        )

                        tempo_total = (
                            fim_atividade
                            -
                            inicio_atividade
                        )

                        delta_t_acc = (
                            x_data_start_absolute[
                                start_50
                            ]
                            -
                            x_data_start_absolute[
                                0
                            ]
                        )

                        if delta_t_acc > 0:

                            ganho_aceleracao = (
                                y_fit_start[
                                    start_50
                                ]
                                /
                                delta_t_acc
                            )

                        else:

                            ganho_aceleracao = np.nan

                        delta_t_dec = (
                            x_data_end_absolute[
                                end_50
                            ]
                            -
                            t_acc[
                                peak_index
                            ]
                        )

                        if delta_t_dec > 0:

                            ganho_desaceleracao = (
                                y_fit_end[
                                    end_50
                                ]
                                /
                                delta_t_dec
                            )

                        else:

                            ganho_desaceleracao = np.nan

                        if (
                            np.isfinite(
                                ganho_aceleracao
                            )
                            and
                            np.isfinite(
                                ganho_desaceleracao
                            )
                            and
                            ganho_desaceleracao != 0
                        ):

                            razao_ganhos = (
                                ganho_aceleracao
                                /
                                ganho_desaceleracao
                            )

                        else:

                            razao_ganhos = np.nan

                        if tempo_total > 0:

                            velocidade_media = (
                                4 / tempo_total
                            )

                        else:

                            velocidade_media = np.nan

                        # =========================================
                        # RESULTADOS
                        # =========================================

                        st.subheader(
                            "Resultados"
                        )

                        c1, c2, c3 = st.columns(
                            3
                        )

                        c1.metric(
                            "Início",
                            f"{inicio_atividade:.2f} s"
                        )

                        c2.metric(
                            "Final",
                            f"{fim_atividade:.2f} s"
                        )

                        c3.metric(
                            "Tempo total",
                            f"{tempo_total:.2f} s"
                        )

                        c1, c2, c3 = st.columns(
                            3
                        )

                        c1.metric(
                            "Amplitude máx. aceleração",
                            f"{amplitude_maxima_aceleracao:.3f} g"
                        )

                        c2.metric(
                            "Amplitude máx. desaceleração",
                            f"{amplitude_maxima_desaceleracao:.3f} g"
                        )

                        c3.metric(
                            "Velocidade média",
                            f"{velocidade_media:.2f} m/s"
                        )

                        c1, c2, c3 = st.columns(
                            3
                        )

                        c1.metric(
                            "Tempo de aceleração",
                            f"{tempo_aceleracao:.2f} s"
                        )

                        c2.metric(
                            "Tempo constante",
                            f"{tempo_constante:.2f} s"
                        )

                        c3.metric(
                            "Tempo de desaceleração",
                            f"{tempo_desaceleracao:.2f} s"
                        )

                        c1, c2, c3 = st.columns(
                            3
                        )

                        c1.metric(
                            "Ganho aceleração",
                            f"{ganho_aceleracao:.4f} g/s"
                        )

                        c2.metric(
                            "Ganho desaceleração",
                            f"{ganho_desaceleracao:.4f} g/s"
                        )

                        c3.metric(
                            "Razão dos ganhos",
                            f"{razao_ganhos:.4f}"
                        )

                    except Exception as e:

                        st.session_state.pop(
                            "inicio_atividade",
                            None
                        )

                        st.session_state.pop(
                            "fim_atividade",
                            None
                        )

                        st.error(
                            "Não foi possível ajustar os modelos."
                        )

                        st.write(
                            f"Motivo: {e}"
                        )

            else:

                st.session_state.pop(
                    "inicio_atividade",
                    None
                )

                st.session_state.pop(
                    "fim_atividade",
                    None
                )

                st.info(
                    "Modelos hiperbólicos desativados."
                )

        except Exception as e:

            st.error(
                f"Erro ao processar acelerômetro: {e}"
            )


# ============================================================
# GIROSCÓPIO
# ============================================================

with tab_gyro:

    st.header(
        "Giroscópio"
    )

    uploaded_gyro = st.file_uploader(
        "Carregue o arquivo do giroscópio",
        type=[
            "txt",
            "csv"
        ],
        key="gyroscope"
    )

    # ========================================================
    # CONFIGURAÇÕES DO FILTRO
    # ========================================================

    st.subheader(
        "Filtro do giroscópio"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        gyro_lowcut = st.number_input(
            "Frequência de corte inferior (Hz)",
            min_value=0.01,
            max_value=49.0,
            value=1.0,
            step=0.1,
            format="%.2f"
        )

    with col2:

        gyro_highcut = st.number_input(
            "Frequência de corte superior (Hz)",
            min_value=0.1,
            max_value=49.0,
            value=4.0,
            step=0.1,
            format="%.2f"
        )

    # ========================================================
    # ESCOLHA DO SINAL
    # ========================================================

    gyro_signal_choice = st.selectbox(
        "Registro do giroscópio a ser analisado",
        options=[
            "X",
            "Y",
            "Z",
            "Norma"
        ],
        index=3
    )

    # ========================================================
    # PARÂMETROS DOS PICOS
    # ========================================================

    st.subheader(
        "Detecção de picos"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        gyro_prominence = st.number_input(
            "Proeminência mínima",
            min_value=0.0,
            value=0.10,
            step=0.01,
            format="%.3f"
        )

    with col2:

        min_peak_distance_seconds = st.number_input(
            "Distância mínima entre picos (s)",
            min_value=0.05,
            max_value=2.0,
            value=0.30,
            step=0.05
        )

    # ========================================================
    # PROCESSAMENTO
    # ========================================================

    if uploaded_gyro is not None:

        try:

            df_gyro = pd.read_csv(
                uploaded_gyro,
                sep=";"
            )

            if df_gyro.shape[
                1
            ] < 4:

                st.error(
                    "O arquivo precisa possuir pelo menos "
                    "4 colunas: tempo, X, Y e Z."
                )

                st.stop()

            (
                t_gyro,
                gyro_x,
                gyro_y,
                gyro_z,
                gyro_norm_raw
            ) = preprocess_sensor(
                df_gyro,
                fs=FS,
                normalize_gravity=False
            )

            # ====================================================
            # FILTRO
            # ====================================================

            gyro_x_filtered = butterworth_filter(
                gyro_x,
                [
                    gyro_lowcut,
                    gyro_highcut
                ],
                FS,
                order=4,
                btype="bandpass"
            )

            gyro_y_filtered = butterworth_filter(
                gyro_y,
                [
                    gyro_lowcut,
                    gyro_highcut
                ],
                FS,
                order=4,
                btype="bandpass"
            )

            gyro_z_filtered = butterworth_filter(
                gyro_z,
                [
                    gyro_lowcut,
                    gyro_highcut
                ],
                FS,
                order=4,
                btype="bandpass"
            )

            # ====================================================
            # NORMA
            # ====================================================

            gyro_norm_filtered = np.sqrt(
                gyro_x_filtered ** 2
                +
                gyro_y_filtered ** 2
                +
                gyro_z_filtered ** 2
            )

            # ====================================================
            # ESCOLHER SINAL
            # ====================================================

            if gyro_signal_choice == "X":

                gyro_signal = (
                    gyro_x_filtered
                )

                ylabel = (
                    "Velocidade angular - X"
                )

            elif gyro_signal_choice == "Y":

                gyro_signal = (
                    gyro_y_filtered
                )

                ylabel = (
                    "Velocidade angular - Y"
                )

            elif gyro_signal_choice == "Z":

                gyro_signal = (
                    gyro_z_filtered
                )

                ylabel = (
                    "Velocidade angular - Z"
                )

            else:

                gyro_signal = (
                    gyro_norm_filtered
                )

                ylabel = (
                    "Norma da velocidade angular"
                )

            # ====================================================
            # GRÁFICO COMPLETO
            # ====================================================

            st.subheader(
                f"Registro selecionado: {gyro_signal_choice}"
            )

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                t_gyro,
                gyro_signal,
                linewidth=1.2
            )

            ax.set_xlabel(
                "Tempo (s)"
            )

            ax.set_ylabel(
                ylabel
            )

            ax.set_title(
                f"Giroscópio - {gyro_signal_choice} "
                f"({gyro_lowcut:.1f}-{gyro_highcut:.1f} Hz)"
            )

            if gyro_signal_choice == "Norma":

                ax.set_ylim(
                    bottom=0
                )

            ax.grid(
                alpha=0.3
            )

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )

            # ====================================================
            # LIMITES DA ATIVIDADE
            # ====================================================

            if (
                "inicio_atividade"
                not in st.session_state
                or
                "fim_atividade"
                not in st.session_state
            ):

                st.warning(
                    "Para detectar os picos, primeiro processe "
                    "o acelerômetro com o modelo hiperbólico ativo."
                )

            else:

                inicio_atividade = (
                    st.session_state[
                        "inicio_atividade"
                    ]
                )

                fim_atividade = (
                    st.session_state[
                        "fim_atividade"
                    ]
                )

                inicio_real = max(
                    inicio_atividade,
                    t_gyro[
                        0
                    ]
                )

                fim_real = min(
                    fim_atividade,
                    t_gyro[
                        -1
                    ]
                )

                if inicio_real >= fim_real:

                    st.error(
                        "O intervalo identificado no acelerômetro "
                        "não coincide com o registro do giroscópio."
                    )

                else:

                    # =============================================
                    # RECORTE
                    # =============================================

                    activity_mask = (
                        (t_gyro >= inicio_real)
                        &
                        (t_gyro <= fim_real)
                    )

                    t_activity = (
                        t_gyro[
                            activity_mask
                        ]
                    )

                    gyro_activity = (
                        gyro_signal[
                            activity_mask
                        ]
                    )

                    # =============================================
                    # DETECÇÃO DE PICOS
                    # =============================================

                    min_distance_samples = max(
                        1,
                        int(
                            min_peak_distance_seconds
                            *
                            FS
                        )
                    )

                    peaks, properties = find_peaks(
                        gyro_activity,
                        prominence=gyro_prominence,
                        distance=min_distance_samples
                    )

                    number_of_peaks = len(
                        peaks
                    )

                    # =============================================
                    # RESULTADOS
                    # =============================================

                    st.subheader(
                        "Resultados"
                    )

                    c1, c2, c3 = st.columns(
                        3
                    )

                    c1.metric(
                        "Início da atividade",
                        f"{inicio_real:.2f} s"
                    )

                    c2.metric(
                        "Final da atividade",
                        f"{fim_real:.2f} s"
                    )

                    c3.metric(
                        "Número de picos",
                        number_of_peaks
                    )

                    if number_of_peaks >= 2:

                        peak_times = (
                            t_activity[
                                peaks
                            ]
                        )

                        peak_intervals = np.diff(
                            peak_times
                        )

                        mean_interval = np.mean(
                            peak_intervals
                        )

                        sd_interval = np.std(
                            peak_intervals,
                            ddof=1
                        ) if len(
                            peak_intervals
                        ) > 1 else 0

                        frequency = (
                            1
                            /
                            mean_interval
                        )

                        c1, c2, c3 = st.columns(
                            3
                        )

                        c1.metric(
                            "Intervalo médio",
                            f"{mean_interval:.3f} s"
                        )

                        c2.metric(
                            "DP dos intervalos",
                            f"{sd_interval:.3f} s"
                        )

                        c3.metric(
                            "Frequência média",
                            f"{frequency:.2f} Hz"
                        )

                    # =============================================
                    # GRÁFICO COMPLETO
                    # =============================================

                    st.subheader(
                        "Registro completo com intervalo analisado"
                    )

                    fig, ax = plt.subplots(
                        figsize=(12, 5)
                    )

                    ax.plot(
                        t_gyro,
                        gyro_signal,
                        linewidth=1.1,
                        label=gyro_signal_choice
                    )

                    ax.axvline(
                        inicio_real,
                        linestyle="--",
                        label="Início"
                    )

                    ax.axvline(
                        fim_real,
                        linestyle="--",
                        label="Final"
                    )

                    ax.axvspan(
                        inicio_real,
                        fim_real,
                        alpha=0.1
                    )

                    ax.scatter(
                        t_activity[
                            peaks
                        ],
                        gyro_activity[
                            peaks
                        ],
                        s=55,
                        marker="o",
                        label=f"Picos (n={number_of_peaks})"
                    )

                    ax.set_xlabel(
                        "Tempo (s)"
                    )

                    ax.set_ylabel(
                        ylabel
                    )

                    ax.set_title(
                        f"Giroscópio - {gyro_signal_choice}"
                    )

                    if gyro_signal_choice == "Norma":

                        ax.set_ylim(
                            bottom=0
                        )

                    ax.grid(
                        alpha=0.3
                    )

                    ax.legend()

                    st.pyplot(
                        fig
                    )

                    plt.close(
                        fig
                    )

                    # =============================================
                    # GRÁFICO APENAS DA ATIVIDADE
                    # =============================================

                    st.subheader(
                        "Intervalo da atividade"
                    )

                    fig, ax = plt.subplots(
                        figsize=(12, 5)
                    )

                    ax.plot(
                        t_activity,
                        gyro_activity,
                        linewidth=1.3
                    )

                    ax.scatter(
                        t_activity[
                            peaks
                        ],
                        gyro_activity[
                            peaks
                        ],
                        s=60,
                        marker="o",
                        label=f"Picos (n={number_of_peaks})"
                    )

                    ax.set_xlabel(
                        "Tempo (s)"
                    )

                    ax.set_ylabel(
                        ylabel
                    )

                    ax.set_title(
                        f"{gyro_signal_choice} durante a atividade"
                    )

                    if gyro_signal_choice == "Norma":

                        ax.set_ylim(
                            bottom=0
                        )

                    ax.grid(
                        alpha=0.3
                    )

                    ax.legend()

                    st.pyplot(
                        fig
                    )

                    plt.close(
                        fig
                    )

                    # =============================================
                    # TABELA
                    # =============================================

                    if number_of_peaks > 0:

                        peak_table = pd.DataFrame(
                            {
                                "Pico":
                                    np.arange(
                                        1,
                                        number_of_peaks + 1
                                    ),

                                "Tempo (s)":
                                    t_activity[
                                        peaks
                                    ],

                                "Amplitude":
                                    gyro_activity[
                                        peaks
                                    ],

                                "Proeminência":
                                    properties[
                                        "prominences"
                                    ]
                            }
                        )

                        with st.expander(
                            "Tabela dos picos"
                        ):

                            st.dataframe(
                                peak_table,
                                use_container_width=True
                            )

                    # =============================================
                    # VISUALIZAÇÃO DE TODOS OS EIXOS
                    # =============================================

                    with st.expander(
                        "Visualizar todos os registros do giroscópio"
                    ):

                        fig, ax = plt.subplots(
                            figsize=(12, 5)
                        )

                        ax.plot(
                            t_gyro,
                            gyro_x_filtered,
                            label="X"
                        )

                        ax.plot(
                            t_gyro,
                            gyro_y_filtered,
                            label="Y"
                        )

                        ax.plot(
                            t_gyro,
                            gyro_z_filtered,
                            label="Z"
                        )

                        ax.axvline(
                            inicio_real,
                            linestyle="--"
                        )

                        ax.axvline(
                            fim_real,
                            linestyle="--"
                        )

                        ax.axvspan(
                            inicio_real,
                            fim_real,
                            alpha=0.1
                        )

                        ax.set_xlabel(
                            "Tempo (s)"
                        )

                        ax.set_ylabel(
                            "Velocidade angular"
                        )

                        ax.set_title(
                            "X, Y e Z do giroscópio"
                        )

                        ax.grid(
                            alpha=0.3
                        )

                        ax.legend()

                        st.pyplot(
                            fig
                        )

                        plt.close(
                            fig
                        )

        except Exception as e:

            st.error(
                f"Erro ao processar giroscópio: {e}"
            )
