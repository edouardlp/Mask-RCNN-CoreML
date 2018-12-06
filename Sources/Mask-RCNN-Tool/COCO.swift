import Foundation

class COCO {
    
    let instances:COCOInstances
    
    struct Index {
        
        let annotationsByImageIds:[Int:[COCOAnnotation]]
        
        static func build(instances:COCOInstances) -> Index {
            
            var annotationsByImageIds:[Int:[COCOAnnotation]] = [:]
            
            for annotation in instances.annotations {
                
                if annotationsByImageIds[annotation.imageId] == nil {
                    annotationsByImageIds[annotation.imageId] = [COCOAnnotation]()
                }
                
                annotationsByImageIds[annotation.imageId]?.append(annotation)
            }
            
            return Index(annotationsByImageIds:annotationsByImageIds)
        }
    }
    
    struct ImageIterator:IteratorProtocol {
        
        let images:[COCOImage]
        var imageIterator:IndexingIterator<[COCOImage]>
        let index:Index
        
        init(images:[COCOImage], index:Index) {
            self.images = images
            self.imageIterator = self.images.makeIterator()
            self.index = index
        }
        
        mutating func next() -> (COCOImage,[COCOAnnotation])? {
            guard let nextImage = self.imageIterator.next()
                else { return nil }
            let annotations = self.index.annotationsByImageIds[nextImage.id] ?? []
            return (nextImage,annotations)
        }
        
    }
    
    lazy var index:Index = {
        return Index.build(instances:self.instances)
    }()
    
    init(url:URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf:url)
        self.instances = try decoder.decode(COCOInstances.self, from: data)
    }
    
    func makeImageIterator(limit:Int? = nil, sortById:Bool = false) -> ImageIterator {
        let images:[COCOImage] = {
            () -> [COCOImage] in
            var sorted = self.instances.images
            if(sortById) {
                sorted = sorted.sorted(by: {
                    l, r in
                    return l.id < r.id
                })
            }
            var limited = sorted
            if let limit = limit {
                limited = Array(limited[0..<limit])
            }
            return limited
        }()
        return ImageIterator(images:images, index:self.index)
    }
}
struct COCOInstances : Codable {
    let info:COCOInfo
    let images:[COCOImage]
    let annotations:[COCOAnnotation]
}

struct COCOInfo : Codable {
    let description:String
    let url:URL
    let version:String
    let year:Int
    let contributor:String
}

struct COCOImage : Codable {
    let id:Int
    let fileName:String
    let width:Int
    let height:Int
    //let cocoURL:URL
}

struct COCOAnnotation : Codable {
    let id:Int
    let imageId:Int
    let categoryId:Int
    //let iscrowd:Bool
    let bbox:[Double]
}
