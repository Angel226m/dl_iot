import { Cpu, Server, Monitor, Users, Building2, Shield, Code, Server as ServerIcon, Package, GraduationCap, Camera, Database, Sparkles, ScanSearch, Eye, Calendar, CheckCircle, Clock } from 'lucide-react'

const Funcionamiento = () => {
  const layers = [
    {
      icon: <Cpu className="w-12 h-12" />,
      title: 'Capa de Hardware',
      technology: 'Raspberry Pi 4 + Arducam',
      color: 'from-green-500 to-emerald-600',
      description: 'Dispositivos IoT para captura de imágenes en tiempo real',
      features: ['Raspberry Pi 4', 'Cámara de alta resolución', 'Sensores ambientales'],
    },
    {
      icon: <Server className="w-12 h-12" />,
      title: 'Capa de Servidor',
      technology: 'Python + Flask + TensorFlow',
      color: 'from-blue-500 to-cyan-600',
      description: 'Procesamiento y análisis con modelos de Deep Learning',
      features: ['API Flask', 'Modelo U-Net', 'Optimización con TensorFlow'],
    },
    {
      icon: <Monitor className="w-12 h-12" />,
      title: 'Capa de Cliente',
      technology: 'React + TypeScript + Tailwind',
      color: 'from-purple-500 to-pink-600',
      description: 'Interfaz web responsiva para visualización de resultados',
      features: ['React 18', 'TypeScript', 'Diseño adaptativo'],
    },
  ]

  const stakeholders = [
    { icon: <Users className="w-10 h-10" />, title: 'Ingenieros Estructurales', description: 'Análisis y mantenimiento de estructuras', color: 'from-blue-500 to-cyan-500' },
    { icon: <Building2 className="w-10 h-10" />, title: 'Propietarios', description: 'Gestión y monitoreo de edificaciones', color: 'from-green-500 to-emerald-500' },
    { icon: <Shield className="w-10 h-10" />, title: 'Autoridades Municipales', description: 'Regulación y cumplimiento de normas', color: 'from-red-500 to-orange-500' },
    { icon: <Code className="w-10 h-10" />, title: 'Equipo de Desarrollo', description: 'Diseño y desarrollo de CrackGuard', color: 'from-purple-500 to-pink-500' },
    { icon: <ServerIcon className="w-10 h-10" />, title: 'Administradores de Servidores', description: 'Mantenimiento de infraestructura IT', color: 'from-yellow-500 to-orange-500' },
    { icon: <Package className="w-10 h-10" />, title: 'Proveedores', description: 'Suministro de hardware y software', color: 'from-cyan-500 to-teal-500' },
    { icon: <GraduationCap className="w-10 h-10" />, title: 'Universidad Nacional de Cañete', description: 'Supervisión académica y evaluación', color: 'from-indigo-500 to-purple-500' },
  ]

  const flowSteps = [
    { icon: <Camera className="w-12 h-12" />, title: 'Captura de Datos', description: 'Raspberry Pi con Arducam captura imágenes de estructuras', number: 1, color: 'from-blue-500 to-cyan-500' },
    { icon: <Database className="w-12 h-12" />, title: 'Entrenamiento del Modelo', description: 'Modelo U-Net entrenado con dataset de grietas', number: 2, color: 'from-green-500 to-emerald-500' },
    { icon: <Sparkles className="w-12 h-12" />, title: 'Extracción de Características', description: 'Red neuronal identifica patrones relevantes', number: 3, color: 'from-purple-500 to-pink-500' },
    { icon: <ScanSearch className="w-12 h-12" />, title: 'Detección y Segmentación', description: 'Algoritmo U-Net delimita áreas con grietas', number: 4, color: 'from-orange-500 to-red-500' },
    { icon: <Eye className="w-12 h-12" />, title: 'Visualización de Resultados', description: 'Interfaz web muestra resultados con métricas', number: 5, color: 'from-pink-500 to-purple-500' },
  ]

  const timeline = [
    { week: 'Semana 1-2', dates: '06 Oct - 19 Oct 2025', activities: ['Investigación y planificación', 'Definición de requisitos', 'Diseño de arquitectura'], responsible: 'Todo el equipo', deliverable: 'Documento de especificaciones' },
    { week: 'Semana 3-4', dates: '20 Oct - 02 Nov 2025', activities: ['Configuración de Raspberry Pi', 'Desarrollo backend Flask', 'Recolección de dataset'], responsible: 'Garay, Figueroa', deliverable: 'Prototipo de hardware y API' },
    { week: 'Semana 5', dates: '03 Nov - 09 Nov 2025', activities: ['Entrenamiento del modelo U-Net', 'Optimización de hiperparámetros', 'Validación del modelo'], responsible: 'Borjas, Conca', deliverable: 'Modelo entrenado' },
    { week: 'Semana 6', dates: '10 Nov - 16 Nov 2025', activities: ['Desarrollo del frontend React', 'Integración con backend', 'Diseño de interfaz'], responsible: 'Figueroa, Garay', deliverable: 'Aplicación web funcional' },
    { week: 'Semana 7', dates: '17 Nov - 23 Nov 2025', activities: ['Pruebas integrales', 'Corrección de bugs', 'Documentación final', 'Preparación de presentación'], responsible: 'Todo el equipo', deliverable: 'Sistema completo y documentación' },
  ]

  return (
    <div className="pt-20">
      {/* Arquitectura de Tres Capas */}
      <section className="bg-gradient-to-br from-blue-50 to-purple-50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-gray-800 mb-4 text-center">🏗️ Arquitectura de Tres Capas</h2>
          <p className="text-xl text-gray-600 text-center max-w-3xl mx-auto mb-12">
            Sistema modular con integración escalable y eficiente
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
            {layers.map((layer, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300">
                <div className={`bg-gradient-to-br ${layer.color} text-white p-6 rounded-lg mb-4 flex justify-center`}>
                  {layer.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-800 text-center mb-2">{layer.title}</h3>
                <p className="text-sm font-semibold text-blue-600 text-center mb-2">{layer.technology}</p>
                <p className="text-gray-600 text-center mb-4">{layer.description}</p>
                <div className="space-y-2">
                  {layer.features.map((feature, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-sm text-gray-700">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span>{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Diagrama de flujo */}
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h3 className="text-2xl font-bold text-center mb-8 text-gray-800">Flujo de Comunicación</h3>
            <div className="flex flex-col md:flex-row items-center justify-center gap-8">
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center text-white font-bold text-2xl">1</div>
                <p className="mt-2 font-medium">Hardware</p>
              </div>
              <div className="text-4xl text-gray-300 rotate-90 md:rotate-0">→</div>
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-full flex items-center justify-center text-white font-bold text-2xl">2</div>
                <p className="mt-2 font-medium">Servidor</p>
              </div>
              <div className="text-4xl text-gray-300 rotate-90 md:rotate-0">→</div>
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center text-white font-bold text-2xl">3</div>
                <p className="mt-2 font-medium">Cliente</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stakeholders */}
      <section className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-gray-800 mb-4 text-center">👥 Actores y Stakeholders</h2>
          <p className="text-xl text-gray-600 text-center max-w-3xl mx-auto mb-12">
            Partes interesadas clave en el desarrollo de CrackGuard
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {stakeholders.map((stakeholder, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300">
                <div className={`bg-gradient-to-br ${stakeholder.color} text-white p-4 rounded-lg mb-4 flex items-center justify-center`}>
                  {stakeholder.icon}
                </div>
                <h3 className="text-lg font-semibold text-gray-800 text-center mb-2">{stakeholder.title}</h3>
                <p className="text-gray-600 text-sm text-center">{stakeholder.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Flujo de Detección */}
      <section className="bg-gradient-to-br from-purple-50 to-pink-50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-gray-800 mb-4 text-center">🔄 Flujo de Detección de Grietas</h2>
          <p className="text-xl text-gray-600 text-center max-w-3xl mx-auto mb-12">
            Proceso completo desde la captura hasta la visualización de resultados
          </p>

          <div className="space-y-8">
            {flowSteps.map((step, index) => (
              <div key={index} className="flex flex-col md:flex-row items-center gap-6">
                <div className={`flex-shrink-0 w-24 h-24 bg-gradient-to-br ${step.color} rounded-full flex items-center justify-center text-white shadow-lg`}>
                  {step.icon}
                </div>
                
                <div className="flex-grow bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`text-3xl font-bold bg-gradient-to-br ${step.color} bg-clip-text text-transparent`}>{step.number}</span>
                    <h3 className="text-2xl font-semibold text-gray-800">{step.title}</h3>
                  </div>
                  <p className="text-gray-600 text-lg">{step.description}</p>
                </div>

                {index < flowSteps.length - 1 && (
                  <div className="hidden md:block text-4xl text-gray-300">↓</div>
                )}
              </div>
            ))}
          </div>

          <div className="mt-12 bg-gradient-to-br from-blue-500 to-purple-600 text-white rounded-xl shadow-2xl p-8">
            <h3 className="text-2xl font-bold text-center mb-6">Deep Learning en Acción</h3>
            <div className="flex flex-col md:flex-row justify-around items-center gap-6">
              <div className="text-center">
                <Database className="w-16 h-16 mx-auto mb-2" />
                <p className="font-semibold">Entrada</p>
                <p className="text-sm">Imágenes</p>
              </div>
              <div className="text-4xl">→</div>
              <div className="text-center">
                <Cpu className="w-16 h-16 mx-auto mb-2" />
                <p className="font-semibold">Procesamiento</p>
                <p className="text-sm">U-Net Model</p>
              </div>
              <div className="text-4xl">→</div>
              <div className="text-center">
                <Eye className="w-16 h-16 mx-auto mb-2" />
                <p className="font-semibold">Salida</p>
                <p className="text-sm">Detección</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Cronograma */}
      <section className="bg-gradient-to-br from-green-50 to-blue-50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-gray-800 mb-4 text-center">📅 Cronograma del Proyecto</h2>
          <p className="text-xl text-gray-600 text-center max-w-3xl mx-auto mb-12">
            Planificación de 7 semanas (06 Oct - 23 Nov 2025)
          </p>

          <div className="space-y-6">
            {timeline.map((item, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300">
                <div className="flex flex-col md:flex-row gap-6">
                  <div className="flex-shrink-0">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold text-xl">
                      {index + 1}
                    </div>
                  </div>

                  <div className="flex-grow">
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-3">
                      <h3 className="text-2xl font-bold text-gray-800">{item.week}</h3>
                      <div className="flex items-center gap-2 text-gray-600">
                        <Calendar className="w-5 h-5" />
                        <span className="font-medium">{item.dates}</span>
                      </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold text-gray-700 mb-2 flex items-center gap-2">
                          <Clock className="w-4 h-4 text-blue-600" />
                          Actividades:
                        </h4>
                        <ul className="space-y-1">
                          {item.activities.map((activity, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-gray-600">
                              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-1" />
                              <span>{activity}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <div className="bg-blue-50 rounded-lg p-4">
                          <h4 className="font-semibold text-gray-700 mb-2">👥 Responsables:</h4>
                          <p className="text-gray-600 mb-3">{item.responsible}</p>
                          
                          <h4 className="font-semibold text-gray-700 mb-2">📦 Entregable:</h4>
                          <p className="text-blue-700 font-medium">{item.deliverable}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

export default Funcionamiento